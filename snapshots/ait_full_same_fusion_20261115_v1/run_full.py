"""AUGUR native-input joint model and identical-rule historical baselines."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"ait_augur_host_entity_20261114_v1"
sys.path.insert(0,str(BASE))
import capacity as c

OUT=ROOT/"output"
ORIGINAL=BASE/"paper_revision_20261114"
WEIGHTS=c.read(BASE/"scorer_lock.json")
REF=c.read(ORIGINAL/"complete.json")
assert len(WEIGHTS)==80 and WEIGHTS.keys()==REF["validation_seals"].keys()


def continuous(base,local):
    return 1.-(1.-base)*(1.-.5*local)


def joint_view(x,h):
    assert np.array_equal(x[:,:811],h[:,:811])
    return np.c_[h,x[:,811:]]


def cutoff(y,s):
    n=np.sort(s[y==0])[::-1]
    cut=float(np.nextafter(n[len(n)//100],np.inf))
    assert np.sum(n>=cut)<=len(n)//100
    return cut


def context_delta(b,x,held,ctx):
    if ctx is None:
        return np.zeros_like(b)
    pc=ctx.predict_proba(np.c_[x,held[:,20:]])[:,1]
    return np.minimum(.15,np.maximum(0.,pc-b))


def score(job,role,ids,y,ex,models,selected):
    tag,seed,fold=job
    entity,ec,host,hc,heads,joint,jc=models
    hx=c.host_view(ex,ids)
    jx=joint_view(ex,hx)
    key=c.tag_key(job)
    w=WEIGHTS[key]["weight"]
    z,g,*_=c.f.local_views(ids,y)
    local=c.f.local_score(heads,z,g)
    ctrl=c.paths(job,ORIGINAL)
    ctrl_seal=c.read(c.pin(ctrl/"validation_seal.json",REF["validation_seals"][key]))
    ref=c.pin(ctrl/f"{role}.npz",ctrl_seal["validation"] if role=="validation" else REF["test_files"][key])
    assert (ctrl_seal["host_anchor"],ctrl_seal["entity_anchor"])==(WEIGHTS[key]["host_seal"],WEIGHTS[key]["entity_seal"])
    out,alf={},{}
    with np.load(ref,allow_pickle=False) as previous:
        np.testing.assert_array_equal(previous["ids"],ids)
        np.testing.assert_array_equal(previous["y"],y)
        for name,model,ctx,m in (("host",host,hc,hx),("entity",entity,ec,ex),("joint",joint,jc,jx)):
            b=model.predict_proba(m)[:,1]
            if name!="joint":
                ref_name="host_cf" if name=="host" else "state_cf"
                np.testing.assert_array_equal(b,previous[ref_name])
            p=continuous(b,local)
            delta=context_delta(p,m,z,ctx)
            alpha=selected[name] if selected is not None else c.promotion(y,p,delta) if ctx is not None else 0.
            out[name+"_continuous"]=p+alpha*delta
            out[name+"_no_context"]=p
            alf[name]=alpha
        out["weighted_continuous"]=(1-w)*out["host_continuous"]+w*out["entity_continuous"]
        out["weighted_no_context"]=(1-w)*out["host_no_context"]+w*out["entity_no_context"]
        out["old_augur"]=(1-w)*previous["host_cf_local"]+w*previous["state_cf_local"]
    return out,alf


def verified_roles(job):
    tag,seed,fold=job
    if tag=="611":
        return c.f.v.load_test(fold)
    ix=c.f.r.role_indices(c.f.original_422["folds"][fold])["test"]
    return c.f.r.IDS[ix],c.f.r.LABELS[ix],c.f.r.features(ix,c.f.TRAIN_422[fold][-1])


def train_and_test(job):
    tag,seed,fold=job
    prefix=f"{tag}_{fold}_{seed}"
    lock=OUT/(prefix+".json")
    vp=OUT/(prefix+".validation.npz")
    tp=OUT/(prefix+".test.npz")
    plan=c.sha(ROOT/"plan.json")
    code=c.sha(Path(__file__))
    if lock.exists():
        z=c.read(lock)
        assert z["plan_sha256"]==plan and z["code_sha256"]==code
        c.pin(vp,z["validation_sha256"]);c.pin(tp,z["test_sha256"])
        return prefix
    train,ids,y,weights,vi,vy,vx,exposure=c.augmented_data(tag,fold)
    ed,es,hd,hs,entity,ec,host,hc,heads=c.models(job)
    assert exposure==es["augmentation"]==hs["exposure"]
    xx=joint_view(train,c.host_view(train,ids))
    j=c.f.base_fit(xx,y,seed,width=1059,weights=weights)
    jc=None
    if tag=="611":
        n=len(train)-sum(exposure["counts"])
        values,*_=c.f.local_views(ids[:n],y[:n])
        jc=c.f.base_fit(np.c_[xx[:n],values[:,20:]],y[:n],seed,width=1112,trees=160)
    path=OUT/(prefix+".joblib")
    tmp=path.with_suffix(".tmp")
    c.joblib.dump((j,jc),tmp,compress=1)
    tmp.replace(path)
    models=(entity,ec,host,hc,heads,j,jc)
    validation,choices=score(job,"validation",vi,vy,vx,models,None)
    c.npz(vp,ids=vi,y=vy,**validation)
    seal=OUT/(prefix+".seal.json")
    c.save(seal,{"alpha":choices,"cuts":{a:cutoff(vy,p) for a,p in validation.items()},
                 "validation_sha256":c.sha(vp),"model_sha256":c.sha(path),"plan_sha256":plan,
                 "source_anchors":{"host":c.sha(hd/"validation_seal.json"),"entity":c.sha(ed/"validation_seal.json")},
                 "config":j.get_params(),"exposure":exposure})
    it,yt,xt=verified_roles(job)
    assert not set(it)&(set(vi)|set(ids))
    test,_=score(job,"test",it,yt,xt,models,choices)
    c.npz(tp,ids=it,y=yt,**test)
    c.save(lock,{"validation_sha256":c.sha(vp),"test_sha256":c.sha(tp),"seal_sha256":c.sha(seal),
                "plan_sha256":plan,"code_sha256":code,"model_sha256":c.sha(path)})
    print("complete",prefix,flush=True)
    return prefix


def main():
    OUT.mkdir(exist_ok=True)
    seeds=list(c.CONFIG["seeds"])
    jobs=[(tag,seed,f"fold_{i:02d}") for i in range(1,9) for tag in ("611","422") for seed in seeds]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("fork")) as pool:
        names=list(pool.map(train_and_test,jobs))
    files=[OUT/(name+f) for name in names for f in (".seal.json",".json",".validation.npz",".test.npz")]
    receipt=OUT/"complete.json"
    c.save(receipt,{"files":{p.name:c.sha(p) for p in files},"cells":len(names),
                    "old_prediction_index":c.sha(ORIGINAL/"complete.json"),
                    "plan_sha256":c.sha(ROOT/"plan.json"),"code_sha256":c.sha(Path(__file__))})
    tmp=ROOT/"output.partial"
    with tarfile.open(tmp,"w:gz") as f:
        for p in [*files,receipt]:
            f.add(p,arcname=p.name)
    tmp.replace(ROOT/"output.tar.gz")


if __name__=="__main__":
    main()


