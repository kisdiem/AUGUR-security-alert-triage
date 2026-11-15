"""Prospective eight-fold train/validation gate; test scoring requires a global seal.

This runner never fits IDF or changes source labels. Features are built independently
by rotation_features.py. All ten primary candidates and six matched-history arms
are retained, including unsupported/no-positive cases and exact source provenance.
"""
from __future__ import annotations
import argparse,collections,concurrent.futures,datetime,hashlib,json,os,pathlib,sys,time,resource
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
DEPS='<RESEARCH_ROOT>/reproductions/nllog_crossscene_422_20261023/deps'
if pathlib.Path(DEPS).is_dir():sys.path.append(DEPS)

SCENES=('fox','harrison','russellmitchell','santos','shaw','wardbeck','wheeler','wilson')
SEEDS=(42,1337,31415,2025,8675309)
ARMS=('none','repeat','cf','norm','norm_repeat')
MAIN=tuple((rep,arm) for rep in ('S0','S3') for arm in ARMS)
HISTORY=tuple((rep,arm) for rep in ('Hcount','Hflat','Hstate') for arm in ('none','cf'))
PARAMS=dict(n_estimators=300,max_depth=6,num_leaves=31,learning_rate=.05,
    is_unbalance=True,colsample_bytree=.8,subsample=.8,subsample_freq=1,
    n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def npz(path,**items):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**items)
    tmp.replace(path)
def root_check(root):
    root=root.resolve()
    assert root.parent==pathlib.Path('<RESEARCH_ROOT>/reproductions') and root.name.startswith('security_completion_')
    return root
def split(fold):
    ix=fold-1;assert 0<=ix<8
    return dict(test=[SCENES[ix]],validation=[SCENES[(ix+1)%8]],
                train=[s for s in SCENES if s not in (SCENES[ix],SCENES[(ix+1)%8])])
def folder(root,fold,rep,arm,seed):return root/f'results/trees/fold_{fold:02d}/{rep}/{arm}/seed_{seed}'
def get_x(z,rep,arm,role):return z[f'X_{"Norm_" if arm.startswith("norm") else ""}{rep}_{role}']

def threshold(y,p,budget=None):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
    assert len(y)==len(p)>0 and np.isfinite(p).all() and np.isin(y,[0,1]).all()
    if not y.sum():return float(np.nextafter(p.max(),np.inf))
    order=np.argsort(-p,kind='stable');s=p[order];v=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(v)[ends];fp=ends+1-tp
    if budget is None:
        f=2*tp/(ends+1+int(y.sum()))
        # Distinct scores are descending: first exact maximum is the highest threshold.
        return float(s[ends[int(np.argmax(f))]])
    valid=np.flatnonzero(fp<=budget)
    if not len(valid) or int(tp[valid].max())==0:return float(np.nextafter(s[0],np.inf))
    k=max(valid,key=lambda i:(int(tp[i]),-int(fp[i]),float(s[ends[i]])))
    return float(s[ends[k]])

def metrics(y,p,cut,ids=None):
    from sklearn.metrics import average_precision_score,precision_recall_curve,auc,roc_auc_score
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
    assert y.ndim==p.ndim==1 and len(y)==len(p)>0 and np.isfinite(p).all()
    assert np.isin(y,[0,1]).all()
    pred=p>=cut;tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());P=int(y.sum());N=len(y)-P
    div=lambda a,b:float(a/b) if b else None
    if P:
        precision,recall,_=precision_recall_curve(y,p)
        ap=float(average_precision_score(y,p));pr=float(auc(recall,precision))
    else:ap=pr=None
    ids=np.asarray(ids if ids is not None else [f'{i:012d}' for i in range(len(y))]).astype(str)
    order=np.lexsort((ids,-p))
    return dict(n=len(y),positive=P,negative=N,threshold=float(cut),AP=ap,PR_AUC_trapezoid=pr,
        ROC_AUC=float(roc_auc_score(y,p)) if P and N else None,
        precision=div(tp,tp+fp),recall=div(tp,P),F1=div(2*tp,2*tp+fp+P-tp),accuracy=div(tp+N-fp,len(y)),
        FPR=div(fp,N),TP=tp,FP=fp,FN=P-tp,TN=N-fp,
        alerts_per_1000_sessions=div(int(pred.sum())*1000,len(y)),
        ranking={str(k):dict(K_effective=min(k,len(y)),TP=int(y[order[:k]].sum()),
            precision=div(int(y[order[:k]].sum()),min(k,len(y))),recall=div(int(y[order[:k]].sum()),P),
            selected_ids=ids[order[:k]].tolist()) for k in (10,20,50)},
        ranking_tie_rule='descending score, ascending stable session ID; labels not used')

def audit_features(root,fold,z):
    roles=split(fold);row_counts={}
    for role in ('train','validation'):
        ids=z['ids_'+role].astype(str);y=z['y_'+role];sc=z['scene_'+role].astype(str)
        assert len(ids)==len(set(ids))==len(y)==len(sc) and np.isin(y,[0,1]).all()
        assert set(sc)==set(roles[role])
        row_counts[role]=dict(n=len(ids),positive=int(y.sum()),scenes=dict(collections.Counter(sc)))
    assert set(z['ids_train']).isdisjoint(z['ids_validation'])
    pi=z['parent_indices'];assert pi.dtype.kind in 'iu' and len(pi)==len(set(pi.tolist()))
    assert np.all((pi>=0)&(pi<len(z['y_train'])))
    np.testing.assert_array_equal(z['ids_train'][pi],z['parent_ids'])
    np.testing.assert_array_equal(z['y_train'][pi],z['cf_y'])
    for rep in ('S0','S3','Hcount','Hflat','Hstate'):
        a=z['X_'+rep+'_train'];b=z['X_'+rep+'_validation'];c=z['CF_'+rep]
        assert a.ndim==b.ndim==c.ndim==2 and a.shape[1]==b.shape[1]==c.shape[1]
        assert len(a)==len(z['y_train']) and len(b)==len(z['y_validation']) and len(c)==len(pi)
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()
    for rep in ('S0','S3'):
        for role in ('train','validation'):
            a=z['X_Norm_'+rep+'_'+role];assert a.shape==z['X_'+rep+'_'+role].shape and np.isfinite(a).all()
    assert z['X_Hflat_train'].shape[1]==z['X_Hstate_train'].shape[1]
    # Hflat and Hstate must expose exactly the same numerical basis.
    n=z['X_Hcount_train'].shape[1]
    for role in ('train','validation'):
        np.testing.assert_array_equal(z['X_Hflat_'+role][:,:n],z['X_Hcount_'+role])
        np.testing.assert_array_equal(z['X_Hstate_'+role][:,:n],z['X_Hcount_'+role])
    np.testing.assert_array_equal(z['CF_Hflat'][:,:n],z['CF_Hcount'])
    np.testing.assert_array_equal(z['CF_Hstate'][:,:n],z['CF_Hcount'])
    return dict(roles=row_counts,parents=len(pi),parent_positive=int(z['cf_y'].sum()),
        dimensions={rep:z['X_'+rep+'_train'].shape[1] for rep in ('S0','S3','Hcount','Hflat','Hstate')})

def train_fold(root,fold):
    from lightgbm import LGBMClassifier
    root=root_check(pathlib.Path(root));source=root/f'data/folds/fold_{fold:02d}/train_validation.npz'
    assert not (root/'PAUSED_BY_USER').exists(),'User pause is active'
    assert not (root/'manifests/global_validation_seal.json').exists(),'No retraining after the global seal in this version'
    z=np.load(source,allow_pickle=False);audit=audit_features(root,fold,z);feature_hash=sha(source);code_hash=sha(__file__)
    save(root/f'manifests/fold_{fold:02d}_runner_input_audit.json',dict(complete=True,**audit,feature_sha256=feature_hash,runner_sha256=code_hash))
    for rep,arm in MAIN+HISTORY:
        X=get_x(z,rep,arm,'train');Y=z['y_train'];Xv=get_x(z,rep,arm,'validation')
        pi=z['parent_indices']
        if arm in ('repeat','norm_repeat','cf'):
            extra=z['CF_'+rep] if arm=='cf' else X[pi]
            Xfit=np.concatenate([X,extra],axis=0);Yfit=np.concatenate([Y,z['cf_y']])
        else:Xfit=X;Yfit=Y
        for seed in SEEDS:
            assert not (root/'PAUSED_BY_USER').exists(),'User pause is active'
            out=folder(root,fold,rep,arm,seed);receipt=out/'validation.json'
            if receipt.exists():
                d=read(receipt)
                assert d['feature_sha256']==feature_hash and d['runner_sha256']==code_hash
                assert sha(out/'model.txt')==d['model_sha256'] and sha(out/'validation_scores.npz')==d['validation_scores_sha256']
                continue
            out.mkdir(parents=True,exist_ok=True);begin=time.time()
            assert np.isin(Yfit,[0,1]).all() and len(np.unique(Yfit))==2
            model=LGBMClassifier(**PARAMS,random_state=seed);model.fit(Xfit,Yfit)
            p=model.predict_proba(Xv)[:,1].astype(np.float64)
            cuts={'max_F1':threshold(z['y_validation'],p)}
            cuts.update({f'FP_{b}':threshold(z['y_validation'],p,b) for b in (0,5,10,20)})
            model.booster_.save_model(str(out/'model.txt'))
            npz(out/'validation_scores.npz',p=p,y=z['y_validation'],ids=z['ids_validation'],scene=z['scene_validation'])
            d=dict(complete=True,phase='validation_only',fold=fold,split=split(fold),representation=rep,arm=arm,seed=seed,
                feature_sha256=feature_hash,runner_sha256=code_hash,model_sha256=sha(out/'model.txt'),
                validation_scores_sha256=sha(out/'validation_scores.npz'),parameters=PARAMS,
                thresholds=cuts,validation={name:metrics(z['y_validation'],p,cut,z['ids_validation']) for name,cut in cuts.items()},
                fit_rows=len(Yfit),real_train_rows=len(Y),appended_rows=len(Yfit)-len(Y),
                fit_positive=int(Yfit.sum()),parent_n=len(pi),cf_coverage=float(len(pi)/len(Y)),
                seconds=time.time()-begin,time=now(),test_accessed=False,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                degenerate_validation_no_positive=not bool(z['y_validation'].sum()))
            save(receipt,d)
            save(root/f'manifests/fold_{fold:02d}_training_progress.json',dict(time=now(),fold=fold,representation=rep,arm=arm,seed=seed,phase='validation_only'))
    select_fold(root,fold)
    return dict(fold=fold,complete=True,tree_models=80)

def select_fold(root,fold):
    rows=[]
    for rank,(rep,arm) in enumerate(MAIN):
        rr=[read(folder(root,fold,rep,arm,seed)/'validation.json') for seed in SEEDS]
        aps=[r['validation']['max_F1']['AP'] for r in rr]
        f1s=[r['validation']['max_F1']['F1'] for r in rr]
        rows.append(dict(representation=rep,arm=arm,priority=rank,
            mean_validation_AP=float(np.mean(aps)) if all(x is not None for x in aps) else None,
            mean_validation_F1=float(np.mean([x if x is not None else 0. for x in f1s])),
            receipt_paths=[str(folder(root,fold,rep,arm,seed)/'validation.json') for seed in SEEDS]))
    eligible=not all(r['mean_validation_AP'] is None for r in rows)
    def choose(candidates):
        if not eligible:return dict(representation='S0',arm='none',valid=False,reason='validation_has_no_positive')
        best=max(candidates,key=lambda r:(r['mean_validation_AP'] if r['mean_validation_AP'] is not None else -float('inf'),r['mean_validation_F1'],-r['priority']))
        return dict(representation=best['representation'],arm=best['arm'],valid=True,mean_validation_AP=best['mean_validation_AP'],mean_validation_F1=best['mean_validation_F1'])
    save(root/f'results/selections/fold_{fold:02d}.json',dict(fold=fold,time=now(),candidate_order=[list(x) for x in MAIN],
        candidates=rows,C_all=choose(rows),C_simple=choose([r for r in rows if r['arm']!='cf']),
        tie_policy='unrounded float64 mean AP, then unrounded mean F1, then fixed priority; exact equality only',
        test_accessed=False,seed_metrics_are_not_deployment_ensemble=True))

def seal(root,require_baselines=True):
    root=root_check(root);files={};trees=0
    for fold in range(1,9):
        for rep,arm in MAIN+HISTORY:
            for seed in SEEDS:
                out=folder(root,fold,rep,arm,seed);d=read(out/'validation.json');assert d['complete'] and d['test_accessed'] is False
                for name,key in (('model.txt','model_sha256'),('validation_scores.npz','validation_scores_sha256')):
                    assert sha(out/name)==d[key]
                for name in ('validation.json','validation_scores.npz','model.txt'):files[(out/name).relative_to(root).as_posix()]=sha(out/name)
                trees+=1
        selection=root/f'results/selections/fold_{fold:02d}.json';assert read(selection)['test_accessed'] is False
        files[selection.relative_to(root).as_posix()]=sha(selection)
        for name in ('train_validation.npz','features_receipt.json','idf_original.json','idf_norm.json'):
            path=root/f'data/folds/fold_{fold:02d}'/name
            assert path.is_file(),path;files[path.relative_to(root).as_posix()]=sha(path)
    assert trees==640
    baseline_counts={};baseline_entries=[]
    if require_baselines:
        for baseline in ('deepcase','erm','groupdro'):
            baseline_counts[baseline]=0
            for fold in range(1,9):
                for seed in SEEDS:
                    out=root/f'results/baselines/{baseline}/fold_{fold:02d}/seed_{seed}'
                    receipt=out/'validation.json'
                    assert receipt.is_file(),f'Missing baseline validation receipt: {receipt}'
                    d=read(receipt);assert d.get('complete',False)
                    assert not d.get('test_accessed',False)
                    entry=read(out/'validation_seal_entry.json')
                    for group in ('files','input_hashes'):
                        for path,h in entry[group].items():
                            p=pathlib.Path(path);assert p.is_file() and sha(p)==h
                            if p.is_relative_to(root):files[p.relative_to(root).as_posix()]=h
                    baseline_entries.append(entry)
                    for p in out.rglob('*'):
                        if p.is_file() and not p.name.startswith('test'):files[p.relative_to(root).as_posix()]=sha(p)
                    baseline_counts[baseline]+=1
    for p in (root/'src').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(root).as_posix()]=sha(p)
    for p in (root/'protocol').rglob('*'):
        if p.is_file():files[p.relative_to(root).as_posix()]=sha(p)
    official_lock=root/'manifests/baseline_source_lock.json'
    if require_baselines:
        assert official_lock.is_file();files[official_lock.relative_to(root).as_posix()]=sha(official_lock)
    # Freeze each scene's immutable source and per-text cache dependencies too;
    # test features must never be rebuilt from a silently changed scene shard.
    for scene in SCENES:
        scene_dir=root/'cache/scenes'/scene
        for name in ('sources_complete.json','features_complete.json','rows.pkl.gz','events.npz'):
            p=scene_dir/name;assert p.is_file(),p
            files[p.relative_to(root).as_posix()]=sha(p)
    for name in ('extra_texts.json','extra_vectors.npy','encoding_complete.json'):
        p=root/'cache/embeddings'/name;assert p.is_file(),p
        files[p.relative_to(root).as_posix()]=sha(p)
    for name in ('resource_policy.json','features_policy.json','io_build_policy.json'):
        p=root/'manifests'/name;assert p.is_file(),p
        files[p.relative_to(root).as_posix()]=sha(p)
    isolation=root/'audits/feature_isolation_audit.json'
    assert read(isolation)['passed'];files[isolation.relative_to(root).as_posix()]=sha(isolation)
    seal_path=root/'manifests/global_validation_seal.json'
    d=dict(sealed=True,all_eight_validation_sealed=True,time=now(),tree_models=trees,baseline_models=baseline_counts,
        all_fold_validation_before_any_batch_test=True,files=files,
        threshold_source='validation only, frozen per model',runner_sha256=sha(__file__),
        complete_plan_baseline_gate=require_baselines,baseline_validation_entries=baseline_entries)
    if seal_path.exists():assert read(seal_path)['files']==files,'Inputs changed after seal'
    else:save(seal_path,d)
    return dict(sealed=True,files=len(files),tree_models=trees,baseline_models=baseline_counts)

def verify_seal(root,fold=None):
    d=read(root/'manifests/global_validation_seal.json');assert d['sealed'] and d['complete_plan_baseline_gate']
    assert sha(__file__)==d['runner_sha256'],'Runner changed after seal'
    for rel,h in d['files'].items():
        if fold is not None and rel.startswith('results/') and f'fold_{fold:02d}/' not in rel:continue
        assert sha(root/rel)==h,f'Frozen input changed: {rel}'
    return d

def score_fold(root,fold):
    from lightgbm import Booster
    root=root_check(pathlib.Path(root));seal_doc=verify_seal(root,fold)
    feature=root/f'data/folds/fold_{fold:02d}/test.npz';z=np.load(feature,allow_pickle=False)
    assert set(z['scene_test'].astype(str))==set(split(fold)['test'])
    ids=z['ids_test'];y=z['y_test'];feature_hash=sha(feature)
    for rep,arm in MAIN+HISTORY:
        X=get_x(z,rep,arm,'test');assert np.isfinite(X).all() and len(X)==len(y)
        for seed in SEEDS:
            out=folder(root,fold,rep,arm,seed);receipt=out/'test.json';val=read(out/'validation.json')
            if receipt.exists():
                prior=read(receipt);assert prior['feature_sha256']==feature_hash and prior['global_seal_sha256']==sha(root/'manifests/global_validation_seal.json');continue
            begin=time.time();model=Booster(model_file=str(out/'model.txt'));p=model.predict(X,num_threads=2).astype(np.float64)
            npz(out/'test_scores.npz',p=p,y=y,ids=ids,scene=z['scene_test'])
            save(receipt,dict(complete=True,time=now(),fold=fold,representation=rep,arm=arm,seed=seed,
                feature_sha256=feature_hash,validation_receipt_sha256=sha(out/'validation.json'),
                global_seal_sha256=sha(root/'manifests/global_validation_seal.json'),
                test={name:metrics(y,p,cut,ids) for name,cut in val['thresholds'].items()},
                thresholds_frozen_before_test=True,test_scores_sha256=sha(out/'test_scores.npz'),seconds=time.time()-begin))
    return dict(fold=fold,tree_models_scored=80)

def selftest():
    y=np.array([0,1,1,0]);p=np.array([.8,.8,.2,.1]);ids=np.array(['b','a','d','c'])
    assert threshold(y,p)==.2
    assert threshold(y,p,0)>.8
    assert threshold(np.zeros(4),p)>.8
    assert threshold(np.array([0,0,1]),np.array([.9,.8,.1]),1)>.9
    m=metrics(y,p,.2,ids);assert (m['TP'],m['FP'],m['FN'],m['TN'])==(2,1,0,1)
    a=metrics(np.zeros(4),p,1,ids);assert a['AP'] is None and a['recall'] is None
    tied=metrics(np.array([0,1]),np.ones(2),1,np.array(['b','a']))
    assert tied['ranking']['10']['selected_ids']==['a','b']
    assert len(MAIN)*len(SEEDS)*8==400 and len(HISTORY)*len(SEEDS)*8==240
    print(json.dumps(dict(passed=True,tests=['tied_threshold_groups','no_positive_fallback','confusion','stable_ranking_ties','training_budget'])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('<RESEARCH_ROOT>/reproductions/security_completion_20261026_v1'))
    ap.add_argument('--stage',choices=('train_validation','seal','test','selftest'),required=True)
    ap.add_argument('--fold',type=int,choices=range(1,9));ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if args.stage=='selftest':selftest();return
    root=root_check(args.root)
    if args.stage=='seal':print(json.dumps(seal(root),ensure_ascii=False));return
    folds=[args.fold] if args.fold else list(range(1,9));fn=train_fold if args.stage=='train_validation' else score_fold
    if len(folds)==1:print(json.dumps(fn(root,folds[0])));return
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(fn,str(root),fold):fold for fold in folds}
        for future in concurrent.futures.as_completed(tasks):print(json.dumps(future.result()),flush=True)

if __name__=='__main__':main()
