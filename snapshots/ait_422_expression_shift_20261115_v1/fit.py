"""Run matched 4/2/2 training edits with host, entity, and joint history."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import sys
import tarfile

import numpy as np

from prepare import FAMILIES, OUT as EDITS, prepare

ROOT = Path(__file__).resolve().parent
PRI = ROOT.parent/"ait_augur_action_contrast_20261115_v1"
sys.path.insert(0,str(PRI))
spec = spec_from_file_location("previous_release_fit",PRI/"fit.py")
frozen = module_from_spec(spec)
spec.loader.exec_module(frozen)
from history_features import views

c=frozen.c
OUT=ROOT/"output"
BEFORE=PRI/"output"
INDEX=c.read(BEFORE/"complete.json")
RELEASE_REGISTRY=c.read(PRI/"selected_model.json")
TRANSFORMS=("host_profile","entity_only_profile","profile_contrast")
REFERENCE=dict(zip(TRANSFORMS,("host_profile","entity_only_profile","111"),strict=True))
TREATMENTS=("dns_only","repeat","edit")


def original(job, ids, y, role):
    prefix = "_".join((job[0],job[2],str(job[1])))
    file=prefix+f".{role}.npz"
    with np.load(c.pin(BEFORE/file,INDEX["files"][file]),allow_pickle=False) as a:
        np.testing.assert_array_equal(ids,a["ids"])
        np.testing.assert_array_equal(y,a["y"])
        return {arm:a[name] for arm,name in REFERENCE.items()}


def alternate(x,ids,pairs):
    available={sid:i for i,sid in enumerate(ids)}
    which=sorted(set(available)&pairs.keys())
    ix=np.asarray([available[sid] for sid in which],dtype=int)
    changed=x[ix].copy()
    np.testing.assert_array_equal(changed[:,384:768],np.stack([pairs[s][0] for s in which])
                                  if which else changed[:,384:768])
    if which:
        changed[:,384:768]=np.stack([pairs[s][1] for s in which])
    np.testing.assert_array_equal(changed[:,:384],x[ix,:384])
    np.testing.assert_array_equal(changed[:,768:],x[ix,768:])
    assert np.all(np.any(changed!=x[ix],axis=1))
    return ix,changed


def scores_for(models, x, ids, q):
    f=views(c.host_view(x,ids),x)
    scores={}
    for key,model in models.items():
        treatment,arm=key.split("/",1)
        p=model.predict_proba(f[arm])[:,1]
        scores[key]=frozen.continuous(p,q)
    return scores


def archive_models(job,prefix,exposure):
    key=c.tag_key(job)
    lock=RELEASE_REGISTRY["models"][key]
    root=PRI.parent
    models={}
    for arm,path,stamp in (("host_profile",root/"ait_history_complement_20261115_v1/output","host"),
                            ("profile_contrast",root/"ait_history_complement_20261115_v2/output","main")):
        info=c.read(c.pin(path/(prefix+".seal.json"),lock[stamp+"_seal"]))
        assert info["exposure"]==exposure
        fit=c.joblib.load(c.pin(path/(prefix+".joblib"),lock[stamp+"_model"]))
        main,ctx=fit[arm]
        assert ctx is None
        models["dns_only/"+arm]=main
    head=c.joblib.load(c.pin(BEFORE/(prefix+".joblib"),lock["control_model"]))["entity_only_profile"]
    assert head[1] is None
    models["dns_only/entity_only_profile"]=head[0]
    return models


def fit_cell(job):
    tag,seed,fold=job
    prefix=f"{tag}_{fold}_{seed}"
    paths={key:OUT/(prefix+suffix) for key,suffix in {
        "validation":".validation.npz","test":".test.npz","seal":".seal.json","model":".joblib",
        "pairs":".pairs.npz","done":".json"}.items()}
    hashes={"plan":c.sha(ROOT/"plan.json"),"code":c.sha(Path(__file__)),
            "source":c.sha(BEFORE/"complete.json"),"edits":c.sha(EDITS/"complete.json")}
    if paths["done"].is_file():
        check=c.read(paths["done"])
        assert check["hashes"]==hashes
        for name in ("validation","test","seal","model","pairs"):
            c.pin(paths[name],check[name])
        return prefix
    tp,ids,y,weights,vi,vy,vx,exposure=c.augmented_data(tag,fold)
    assert weights is None
    _,es,_,hs,*_,local_heads=c.models(job)
    assert exposure==es["augmentation"]==hs["exposure"]
    n=len(ids)-exposure["parents"]
    by_family=PAIRS[fold]
    matches={family:alternate(tp[:n],ids[:n],bank) for family,bank in by_family.items()}
    fit_ids=np.r_[ids,*[ids[ix] for ix,_ in matches.values()]]
    yy=np.r_[y,*[y[ix] for ix,_ in matches.values()]]
    assert set(fit_ids).issubset(ids[:n])
    models=archive_models(job,prefix,exposure)
    for treatment in ("repeat","edit"):
        matrix=np.concatenate([tp,*[xt if treatment=="edit" else tp[ix] for ix,xt in matches.values()]])
        v=views(c.host_view(matrix,fit_ids),matrix)
        for arm in TRANSFORMS:
            x=v[arm]
            models[treatment+"/"+arm]=c.f.base_fit(x,yy,seed,width=x.shape[1])
    probe_v, gate,*_=c.f.local_views(vi,vy)
    qv=c.f.local_score(local_heads,probe_v,gate)
    validation=scores_for(models,vx,vi,qv)
    ref=original(job,vi,vy,"validation")
    for key,value in ref.items():
        np.testing.assert_array_equal(validation["dns_only/"+key],value)
    c.npz(paths["validation"],ids=vi,y=vy,**validation)
    temp=paths["model"].with_suffix(".tmp")
    c.joblib.dump({k:v for k,v in models.items() if not k.startswith("dns_only/")},temp,compress=1)
    temp.replace(paths["model"])
    c.save(paths["seal"],{"hashes":hashes,"validation":c.sha(paths["validation"]),"model":c.sha(paths["model"]),
                         "parents":{k:{"total":len(ix),"positive":int(y[ix].sum())} for k,(ix,_) in matches.items()},
                         "thresholds":{k:frozen.cutoff(vy,p) for k,p in validation.items()}})

    ti,ty,tx=frozen.verified_roles(job)
    assert not set(fit_ids)&(set(ti)|set(vi)) and not set(vi)&set(ti)
    local,gate,*_=c.f.local_views(ti,ty)
    qt=c.f.local_score(local_heads,local,gate)
    test=scores_for(models,tx,ti,qt)
    ref=original(job,ti,ty,"test")
    for key,value in ref.items():
        np.testing.assert_array_equal(test["dns_only/"+key],value)
    c.npz(paths["test"],ids=ti,y=ty,**test)
    probes={}
    for fam,bank in by_family.items():
        ix,edited=alternate(tx,ti,bank)
        probes[fam+"/positions"]=ix
        if len(ix):
            for name,alt in scores_for(models,edited,ti[ix],qt[ix]).items():
                probes[fam+"/"+name]=alt
    c.npz(paths["pairs"],**probes)
    c.save(paths["done"],{"hashes":hashes,**{k:c.sha(p) for k,p in paths.items() if k!="done"}})
    print("complete",prefix,flush=True)
    return prefix


def main():
    global PAIRS
    sealed=prepare(frozen)
    PAIRS={}
    for fold in c.f.original_422["folds"]:
        pools={}
        for family in FAMILIES:
            name=fold+"_"+family+".npz"
            with np.load(c.pin(EDITS/name,sealed["files"][name]),allow_pickle=False) as f:
                pools[family]=dict(zip(f["ids"],zip(f["original"],f["alternate"],strict=True),strict=True))
        PAIRS[fold]=pools
    OUT.mkdir(exist_ok=True)
    jobs=[("422",seed,f"fold_{i:02d}") for i in range(1,9) for seed in c.CONFIG["seeds"]]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AUGUR_CPU_WORKERS","6"))) as pool:
        names=list(pool.map(fit_cell,jobs))
    files=[OUT/(p+s) for p in names for s in (".json",".seal.json",".pairs.npz",".validation.npz",".test.npz")]
    complete=OUT/"complete.json"
    c.save(complete,{"files":{p.name:c.sha(p) for p in files},"cells":len(names),"plan":c.sha(ROOT/"plan.json"),
                     "code":c.sha(Path(__file__)),"edits":c.sha(EDITS/"complete.json"),
                     "source":c.sha(BEFORE/"complete.json")})
    tmp=ROOT/"output.partial"
    with tarfile.open(tmp,"w:gz") as f:
        for p in [*files,complete]:
            f.add(p,arcname=p.name)
        f.add(EDITS/"complete.json",arcname="edit_manifest.json")
    tmp.replace(ROOT/"output.tar.gz")


if __name__=="__main__":
    main()

