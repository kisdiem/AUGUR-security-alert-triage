from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import mean
import json
import sys

import joblib
import numpy as np
from sklearn.metrics import average_precision_score as ap

root=Path(__file__).resolve().parent
source=root.parent/"ait611_cf_context_20261113_v1"
spec=spec_from_file_location("v1", source/"fit_context.py")
v1=module_from_spec(spec)
spec.loader.exec_module(v1)
out=root/"results"
protocol=root/"protocol.json"
previous=v1.c.read(source/"results"/"summary.json")
seals=previous["validation_seals"]
old_models=["weight_cf","context"]
arms=("three","weighted")
variants=("anchor","bounded","corroborated")
frozen="frozen_dns"

def records(fold):
    folder=source/"results"/fold
    receipt=v1.c.read(v1.c.pinned(folder/"validation_seal.json",seals[fold]))
    models={name:joblib.load(v1.c.pinned(folder/(name+".joblib"),receipt["models"][name])) for name in old_models}
    return folder,receipt,models

def inputs(fold,role,record):
    folder,receipt,models=record
    if role=="validation":
        *_,ids,y,x=v1.load_training(fold)
        archived=v1.c.pinned(folder/"validation.npz",receipt["sha256"])
    else:
        ids,y,x=v1.load_test(fold)
        complete=v1.c.read(folder/"complete.json")
        assert complete["validation_seal_sha256"]==seals[fold]
        archived=v1.c.pinned(folder/"test.npz",complete["sha256"])
    dns,three,old_local,_=v1.source_scores(fold,role,ids,y,x)
    xx,_=v1.context(ids,y,x)
    ctx=models["context"].predict_proba(xx)[:,1]
    weighted=models["weight_cf"].predict_proba(x)[:,1]
    mains={"three":three,"weighted":weighted}
    base={k:np.maximum(p,.5*old_local) for k,p in mains.items()}
    base[frozen]=np.maximum(dns,.5*old_local)
    with np.load(archived,allow_pickle=False) as before:
        np.testing.assert_array_equal(ids,before["ids"])
        np.testing.assert_array_equal(y,before["y"])
        for k,other in (("three","three_cf"),("weighted","weighted_cf"),(frozen,frozen)):
            np.testing.assert_array_equal(base[k],before[other])
    return ids,y,mains,base,ctx

def fusion(base,ctx,eligible,weight):
    return base+weight*np.minimum(.15,np.maximum(0,ctx-base))*eligible

def score(y,p,cut):
    predicted=p>=cut
    pos=y==1
    return dict(AP=float(ap(y,p)),recall=float(predicted[pos].mean()),FP=int(predicted[~pos].sum()),
                TP=int(predicted[pos].sum()),negative=int((~pos).sum()))

def validate(fold):
    ids,y,mains,base,ctx=inputs(fold,"validation",records(fold))
    dst=out/fold
    dst.mkdir()
    settings={}
    preds={frozen:base[frozen]}
    cuts={frozen:v1.cut(y,base[frozen])}
    for a in arms:
        p=base[a]
        ap0=float(ap(y,p))
        recall0=score(y,p,v1.cut(y,p))["recall"]
        m_gate=float(np.quantile(mains[a][y==0],.90))
        c_gate=float(np.quantile(ctx[y==0],.95))
        for variant in variants:
            name=a+"_"+variant
            eligible=((mains[a]>m_gate)&(ctx>c_gate)) if variant=="corroborated" else np.ones(len(y),bool)
            options=(0.,) if variant=="anchor" else (0.,.25,.5,1.)
            candidates=[]
            for weight in options:
                q=fusion(p,ctx,eligible,weight)
                t=v1.cut(y,q)
                result=score(y,q,t)
                if result["AP"]>=ap0-1e-12 and result["recall"]>=recall0-1e-12:
                    candidates.append((result["AP"],result["recall"],-weight,q,t))
            _,_,negative_weight,q,t=max(candidates,key=lambda item:item[:3])
            preds[name]=q
            cuts[name]=t
            settings[name]={"weight":-negative_weight,"main_gate":m_gate,"context_gate":c_gate}
    np.savez_compressed(dst/"validation.npz",ids=ids,y=y,**preds)
    path=dst/"validation_seal.json"
    v1.write(path,dict(settings=settings,thresholds=cuts,validation_sha256=v1.sha(dst/"validation.npz"),
                       protocol_sha256=v1.sha(protocol),source_seal_sha256=seals[fold]))
    return fold,v1.sha(path)

def evaluate(item):
    fold,seal=item
    dst=out/fold
    cfg=v1.c.read(v1.c.pinned(dst/"validation_seal.json",seal))
    assert cfg["protocol_sha256"]==v1.sha(protocol)
    assert cfg["source_seal_sha256"]==seals[fold]
    ids,y,mains,base,ctx=inputs(fold,"test",records(fold))
    with np.load(v1.c.pinned(dst/"validation.npz",cfg["validation_sha256"]),allow_pickle=False) as z:
        assert set(ids).isdisjoint(z["ids"])
        assert all(int((z[k][z["y"]==0]>=t).sum())<=int((z["y"]==0).sum())//100 for k,t in cfg["thresholds"].items())
    preds={frozen:base[frozen]}
    output={}
    for a in arms:
        for v in variants:
            name=a+"_"+v
            rule=cfg["settings"][name]
            g=((mains[a]>rule["main_gate"])&(ctx>rule["context_gate"])) if v=="corroborated" else np.ones(len(y),bool)
            preds[name]=fusion(base[a],ctx,g,rule["weight"])
            output[name]=score(y,preds[name],cfg["thresholds"][name])|dict(promoted=int((preds[name]>base[a]).sum()),weight=rule["weight"])
    output[frozen]=score(y,preds[frozen],cfg["thresholds"][frozen])
    p=dst/"test.npz"
    np.savez_compressed(p,ids=ids,y=y,**preds)
    v1.write(dst/"complete.json",dict(test_sha256=v1.sha(p),validation_seal_sha256=seal))
    return v1.cfg["folds"][fold]["scene"],output

def main():
    out.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        validation_seals=dict(pool.map(validate,v1.folds))
        v1.write(out/"all_validation_sealed.json",validation_seals)
        print("all validation rules and thresholds sealed",flush=True)
        scenes=dict(pool.map(evaluate,validation_seals.items()))
    overall={}
    draw=np.random.default_rng(20261113).integers(0,len(scenes),(10000,len(scenes)))
    for k in next(iter(scenes.values())):
        values=[r[k] for r in scenes.values()]
        base=frozen if k==frozen else k.split("_",1)[0]+"_anchor"
        row=dict(AP=mean(r["AP"] for r in values),recall=mean(r["recall"] for r in values),
                 FP=sum(r["FP"] for r in values),TP=sum(r["TP"] for r in values),promoted=sum(r.get("promoted",0) for r in values))
        for other,label in ((frozen,"frozen"),(base,"anchor")):
            ds=np.array([r[k]["AP"]-r[other]["AP"] for r in scenes.values()])
            row[label]=dict(delta=float(ds.mean()),wins=int((ds>1e-12).sum()),
                            ties=int((np.abs(ds)<=1e-12).sum()),ci=np.quantile(ds[draw].mean(1),[.025,.975]).tolist())
        overall[k]=row
    v1.write(out/"summary.json",dict(overall=overall,by_scene=scenes,validation_seals=validation_seals))
    print("complete",flush=True)
if __name__=="__main__":
    main()
