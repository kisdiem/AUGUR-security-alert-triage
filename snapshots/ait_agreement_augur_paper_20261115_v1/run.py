"""Publish matched AUGUR agreement ablations from pinned AIT model artifacts."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import tarfile

import numpy as np

from agreement import agreement

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
RELEASE = BASE / "ait_entity_roles_paper_20261115_v1"
spec = spec_from_file_location("role_release", RELEASE / "run.py")
source = module_from_spec(spec)
spec.loader.exec_module(source)
c, f = source.c, source.f
LEGACY = source.OLD
EXP = source.EXP
ROLES = RELEASE / "output"
OUT = ROOT / "output"
EXP_INDEX = source.direct.INDEX
INDICES = {path:c.read(path/"complete.json") for path in (LEGACY, EXP/"output", ROLES)}
PINS = {k:c.sha(v) for k,v in {
    "plan":ROOT/"plan.json", "run":ROOT/"run.py", "agreement":ROOT/"agreement.py",
    "roles":ROLES/"complete.json", "legacy":LEGACY/"complete.json",
    "edit":EXP/"output/complete.json", "history":source.FEATURES/"complete.json",
}.items()}
FLAGS = ("100","101","110","111")
ARMS = ("000","001","010","011",*FLAGS,"host_profile","entity_only_profile","current_only",
        "previous_augur","redesigned_joint")


def load_models(path,prefix):
    name = prefix + ".seal.json"
    seal = c.read(c.pin(path/name,INDICES[path]["files"][name]))
    return c.joblib.load(c.pin(path/(prefix+".joblib"),seal["model"]))


def get_scores(path,prefix,role,ids,y):
    return source.read_scores(path,INDICES[path],prefix,role,ids,y)


def old_matrix(x,ids):
    v=source.direct.views(c.host_view(x,ids),x)
    return v["host_profile"],v["profile_contrast"]


def fit(job):
    tag,seed,fold=job
    prefix=f"{tag}_{fold}_{seed}"
    artifact=OUT/(prefix+".joblib")
    seal=OUT/(prefix+".seal.json")
    validation=OUT/(prefix+".validation.npz")
    if seal.is_file():
        prev=c.read(seal)
        assert prev["pins"]==PINS
        c.pin(artifact,prev["model"])
        c.pin(validation,prev["validation"])
        return prefix
    _,repeat,ids,y,w,vi,vy,vx,_=source.augment(tag,fold)
    h,j=old_matrix(repeat,ids)
    heads={"host_repeat":c.f.base_fit(h,y,seed,width=h.shape[1],weights=w)}
    if tag=="422":
        heads["joint_repeat"]=c.f.base_fit(j,y,seed,width=j.shape[1],weights=w)
    cache=source_models(tag,prefix,heads)
    predicted,alpha=score(job,"validation",vi,vy,vx,cache,None)
    c.npz(validation,ids=vi,y=vy,**predicted)
    tmp=artifact.with_suffix(".partial")
    c.joblib.dump(heads,tmp,compress=1)
    tmp.replace(artifact)
    c.save(seal,{"pins":PINS,"validation":c.sha(validation),"model":c.sha(artifact),
                 "alpha":alpha,"cutoff":{key:f.cutoff(vy,p) for key,p in predicted.items()}})
    print("sealed",prefix,flush=True)
    return prefix


def source_models(tag,prefix,repeat_heads):
    roles_host=load_models(ROLES,prefix)["host_profile"]
    expr=load_models(EXP/"output",prefix)["edit/profile_contrast"] if tag=="422" else None
    return repeat_heads,roles_host,expr


def score(job,role,ids,y,x,models,alpha):
    tag,seed,fold=job
    prefix=f"{tag}_{fold}_{seed}"
    h,j=old_matrix(x,ids)
    base=get_scores(ROLES,prefix,role,ids,y)
    repeat,(_,host_head,host_context),old_cf=models
    local,gates,*_=c.f.local_views(ids,y)
    q=c.f.local_score(c.models(job)[-1],local,gates)
    host_repeat=repeat["host_repeat"].predict_proba(h)[:,1]
    host_repeat_local=f.continuous(host_repeat,q)
    host_residual=f.context_delta(host_repeat_local,h,local,host_context)
    if alpha is None:
        alpha=c.promotion(y,host_repeat_local,host_residual) if host_context is not None else 0.
    host_repeat_local+=alpha*host_residual
    host={k:p for k,p in zip(FLAGS,(host_repeat,host_repeat_local,host_head.predict_proba(h)[:,1],
                                   base["host_profile"]),strict=True)}
    if tag=="611":
        prior=get_scores(LEGACY,prefix,role,ids,y)
        old={k:prior[k] for k in FLAGS}
    else:
        p_repeat=repeat["joint_repeat"].predict_proba(j)[:,1]
        p_cf=old_cf.predict_proba(j)[:,1]
        old=dict(zip(FLAGS,(p_repeat,f.continuous(p_repeat,q),p_cf,f.continuous(p_cf,q)),strict=True))
    np.testing.assert_array_equal(old["111"],base["previous_augur"])
    observed,relations=np.asarray([SUPPORT[s] for s in ids],dtype=bool).T
    out={k:base[k] for k in ARMS if k not in FLAGS and k!="redesigned_joint"}
    out["redesigned_joint"]=base["111"]
    for a in FLAGS:
        out[a]=agreement(old[a],host[a],base[a],observed,relations)
    return out,alpha


def test(job):
    tag,seed,fold=job
    prefix=f"{tag}_{fold}_{seed}"
    lock=OUT/"validation_complete.json"
    path=OUT/(prefix+".test.npz")
    done=OUT/(prefix+".done.json")
    stamp=c.sha(lock)
    sealed=c.read(c.pin(OUT/(prefix+".seal.json"),c.read(lock)["validation"][prefix]))
    if done.is_file():
        old=c.read(done)
        assert old["lock"]==stamp
        c.pin(path,old["test"])
        return
    ids,y,x=f.verified_roles(job)
    _,train,_,_,vi,*_=c.augmented_data(tag,fold)
    assert not set(ids)&(set(train)|set(vi))
    heads=c.joblib.load(c.pin(OUT/(prefix+".joblib"),sealed["model"]))
    predictions,_=score(job,"test",ids,y,x,source_models(tag,prefix,heads),sealed["alpha"])
    c.npz(path,ids=ids,y=y,**predictions)
    c.save(done,{"lock":stamp,"test":c.sha(path)})
    print("tested",prefix,flush=True)


def main():
    global SUPPORT
    SUPPORT={}
    for scene,h in c.read(source.FEATURES/"complete.json").items():
        folder=source.FEATURES/scene
        info=c.read(c.pin(folder/"complete.json",h))
        with np.load(c.pin(folder/"readout.npz",info["feature"]),allow_pickle=False) as z:
            flags=zip(np.any(z["roles"]>0,axis=1),np.any(z["links"]>0,axis=1),strict=True)
            SUPPORT.update(zip(z["ids"],flags,strict=True))
    assert len(SUPPORT)==11482
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AUGUR_CPU_WORKERS","8"))) as pool:
        names=list(pool.map(fit,source.JOBS))
        c.save(OUT/"validation_complete.json",{"pins":PINS,"validation":{n:c.sha(OUT/(n+".seal.json")) for n in names}})
        list(pool.map(test,source.JOBS))
    files=[OUT/(n+s) for n in names for s in (".test.npz",".validation.npz",".seal.json",".done.json")]
    files.append(OUT/"validation_complete.json")
    receipt=OUT/"complete.json"
    c.save(receipt,{"pins":PINS,"files":{p.name:c.sha(p) for p in files},"cells":len(names)})
    tmp=ROOT/"output.partial"
    with tarfile.open(tmp,"w:gz") as bundle:
        for p in [*files,receipt]:
            bundle.add(p,arcname=p.name)
    tmp.replace(ROOT/"output.tar.gz")


if __name__=="__main__":
    main()

