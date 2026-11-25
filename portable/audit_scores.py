"""Replay frozen scores, validation cutoffs and record-budget queues.

AP / Recall@1% FPR are test-ranking metrics. Operational cutoffs are obtained
only from the validation partition. Repeated held-out appearances are retained.
"""
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
from io import BytesIO,StringIO
from zipfile import ZipFile
import argparse,csv,json,sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from recompute_ait_scores import metrics
from current.score_fusion import agreement
from portable.model import validation_cutoff,digest

def run(archive,output,kind):
    archive=Path(archive);output=Path(output)
    expected=json.loads(archive.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if digest(archive)!=expected['archive_sha256']:raise ValueError('Score archive identity mismatch')
    arms=('correction_off','augur') if kind=='switch' else ('current','host_reference','augur')
    rows=defaultdict(list);counts=defaultdict(lambda:np.zeros(4,dtype=np.float64))
    queue_totals=defaultdict(lambda:np.zeros(4,dtype=np.float64));queue_members={};unique_hits=defaultdict(set)
    detail=[];cutoff_count=0;max_formula_error=0.
    with ZipFile(archive) as z:
        costs={r['id']:(int(r['y']),int(r['alerts'])) for r in csv.DictReader(StringIO(z.read('session_costs.csv').decode('utf-8-sig')))}
        require_cells={f'{p}_fold_{f:02d}_{seed}' for p in ('611','422') for f in range(1,9) for seed in (42,1337,31415,2025,8675309)}
        names={n[len('cells/'):-len('.test.npz')] for n in z.namelist() if n.startswith('cells/') and n.endswith('.test.npz')}
        if names!=require_cells:raise ValueError('Expected exactly 80 sealed cells')
        for prefix in sorted(names):
            protocol,_,fold,seed=prefix.split('_');seed=int(seed)
            with np.load(BytesIO(z.read('cells/'+prefix+'.validation.npz')),allow_pickle=False) as v, np.load(BytesIO(z.read('cells/'+prefix+'.test.npz')),allow_pickle=False) as t:
                if set(v['ids'])&set(t['ids']):raise ValueError('Validation/test overlap')
                for d in (v,t):
                    if len(set(d['ids']))!=len(d['ids']) or not np.isin(d['y'],[0,1]).all():raise ValueError('Invalid ids/labels')
                    if kind=='switch':
                        on=agreement(d['anchor'],d['host'],d['direct'],d['role'],d['relation'])
                        off=d['anchor']+.2*(d['host']-d['anchor'])
                        np.testing.assert_array_equal(on,d['augur']);np.testing.assert_array_equal(off,d['correction_off'])
                        max_formula_error=max(max_formula_error,float(np.max(np.abs(on-d['augur']))))
                y=t['y'];ids=t['ids'];scenes=np.array([s.split('_',1)[0] for s in ids])
                if len(np.unique(scenes))!=(1 if protocol=='611' else 2):raise ValueError('Scene allocation')
                cc=np.array([costs[str(s)][1] for s in ids],dtype=np.int64)
                np.testing.assert_array_equal(y,[costs[str(s)][0] for s in ids])
                sealed=json.loads(z.read('cells/'+prefix+'.cutoffs.json'))
                for arm in arms:
                    cutoff=validation_cutoff(v['y'],v[arm]);cutoff_count+=1
                    if arm in sealed and cutoff!=sealed[arm]:raise ValueError('Validation cutoff mismatch')
                    p=t[arm]
                    if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError('Score shape/nonfinite')
                    called=p>=cutoff
                    counts[protocol,seed,arm]+=np.array([(called&(y==1)).sum(),(called&(y==0)).sum(),(y==1).sum(),(y==0).sum()])
                    for scene in np.unique(scenes):
                        m=scenes==scene;ap,rec=metrics(y[m],p[m])
                        vector=np.array([ap,rec,called[m][y[m]==1].mean(),called[m][y[m]==0].mean()])
                        rows[protocol,seed,str(scene),arm].append(vector)
                        detail.append(dict(protocol=protocol,fold=fold,seed=seed,scene=str(scene),arm=arm,AP=ap,recall_1pct_fpr=rec))
                    allowance=int(cc.sum())*2//100;spent=0;selected=[]
                    for i in np.lexsort((ids,-p)):
                        if spent+int(cc[i])<=allowance:selected.append(int(i));spent+=int(cc[i])
                    selected=np.asarray(selected,dtype=int);tp=int(y[selected].sum());fp=len(selected)-tp
                    queue_totals[protocol,seed,arm]+=np.array([tp,fp,spent,allowance])
                    selected_ids=[str(s) for s in ids[selected]]
                    queue_members[prefix+'/'+arm]=selected_ids
                    unique_hits[protocol,seed,arm].update(str(ids[i]) for i in selected if y[i]==1)
    summary={}
    for protocol in ('611','422'):
        summary[protocol]={}
        for arm in arms:
            scenes=sorted({k[2] for k in rows if k[0]==protocol});seeds=sorted({k[1] for k in rows if k[0]==protocol})
            if len(scenes)!=8 or len(seeds)!=5:raise ValueError('Eight scenes/five seeds required')
            if any(len(rows[protocol,seed,scene,arm])!=(1 if protocol=='611' else 2) for scene in scenes for seed in seeds):raise ValueError('Repeated appearances')
            matrix=np.array([[np.mean(rows[protocol,seed,scene,arm],axis=0) for seed in seeds] for scene in scenes])
            mean=matrix.mean((0,1));seed_values=matrix.mean(0)
            summary[protocol][arm]=dict(AP=float(mean[0]),recall_1pct_fpr=float(mean[1]),
                scene_macro_frozen_recall=float(mean[2]),scene_macro_frozen_negative_rate=float(mean[3]),
                AP_seed_sample_sd=float(seed_values[:,0].std(ddof=1)),
                frozen_counts_per_seed=np.mean([counts[protocol,s,arm] for s in seeds],axis=0).tolist(),
                record_queue_per_seed=np.mean([queue_totals[protocol,s,arm] for s in seeds],axis=0).tolist(),
                unique_positive_sessions_per_seed=float(np.mean([len(unique_hits[protocol,s,arm]) for s in seeds])),
                scene_AP={scene:float(v) for scene,v in zip(scenes,matrix[:,:,0].mean(1),strict=True)})
            ref=expected['expected_metrics'][protocol][arm]
            np.testing.assert_allclose(mean[:2],[ref['AP'],ref['recall_1pct_fpr']],rtol=0,atol=1e-12)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per_scene_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    members=json.dumps(queue_members,sort_keys=True,separators=(',',':')).encode()
    (output/'ordered_record_queues.json').write_bytes(members)
    report=dict(status='PASS',kind=kind,cells=80,array_files=160,validation_cutoffs=cutoff_count,
                maximum_formula_error=max_formula_error,results=summary,
                ordered_queue_sha256=sha256(members).hexdigest(),
                scope='Frozen score-to-metric and deterministic queue replay; not training or independent label verification. Queue digests describe reconstructed membership, not an unavailable historical ordered-list receipt.')
    (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--kind',required=True,choices=['switch','matched']);a=p.parse_args();r=run(a.archive,a.output_dir,a.kind)
    print(json.dumps({k:r[k] for k in ['status','kind','cells','array_files','validation_cutoffs','maximum_formula_error']},indent=2))
if __name__=='__main__':main()
