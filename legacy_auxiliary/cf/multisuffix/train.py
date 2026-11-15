"""Compare equally weighted suffix renderings on the fixed AIT 4/2/2 split."""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    dst=a.root
    base=dst.parent
    cf=base/"反事实实体历史试验_20261105_v1"
    sys.path.insert(0,str(cf/"src"))
    from previous_run import load_role,sha256
    from run_cf import append,operating_points,result_at,top_budgets
    source=base/"实体关联历史试验_20261105_v1"/"data"
    receipt=json.loads((source/"train_validation.json").read_text(encoding="utf-8"))
    assert sha256(source/"train_validation.npz")==receipt["train_validation_sha256"]
    archived=base/"AIT完整基线消融_20261105_v1"/"fixed_422"
    old_seal=json.loads((archived/"validation_seal.json").read_text(encoding="utf-8"))
    cfg=old_seal["protocol"]["config"]
    seeds=old_seal["protocol"]["seeds"]
    train,valid=(load_role(source,"train_validation.npz",role,None) for role in ("train","validation"))
    tr,va=[append(role["state"],role["history"],"host10") for role in (train,valid)]
    with np.load(dst/"views.npz",allow_pickle=False) as z:
        ids,original,views=z["ids"],z["original"],z["variants"]
        n=int(z["train_size"])
    assert len(set(ids))==len(ids) and n<len(ids)
    train_ix,valid_ix=([list(role["ids"]).index(sid) for sid in ids[i:j]]
                       for role,i,j in ((train,0,n),(valid,n,len(ids))))
    original_views=np.vstack((tr[train_ix,:768],va[valid_ix,:768]))
    np.testing.assert_allclose(original_views,original,rtol=2e-5,atol=3e-6)
    y=train["y"]
    assert y[train_ix].sum()*2==n
    nviews=views.shape[1]-1
    assert nviews==4
    augmented=np.r_[y,np.repeat(y[train_ix],nviews)]
    weights=np.ones(len(augmented),dtype=np.float64)
    weights[train_ix]=0.5
    weights[len(y):]=0.5/nviews
    weights/=weights.mean()
    parent=np.repeat(tr[train_ix,None,:],nviews,axis=1)
    vparent=np.repeat(va[valid_ix,None,:],views.shape[1],axis=1)
    var_delta=(views-original[:,None,:]).astype(np.float32)
    vparent[:,:,:768]+=var_delta[n:]
    result_dir=dst/"fixed_422"
    result_dir.mkdir(exist_ok=False)

    protocol=dict(
        split="AIT-ADS fixed 4/2/2, previously examined test",seeds=seeds,config=cfg,
        original_train=len(tr),parent_pairs=n,parents_by_class=[int((y[train_ix]==z).sum()) for z in (0,1)],
        rendering_options=["old two-suffix swap","com","net","org","info"],
        one_current_alert_per_parent=True,heldout_pairs_label_blind=True,
        train_inputs="Four appended views for every selected parent in all matched arms. "
                     "Original plus four variants share one parent weight.",
        target_recall=.80,top_k=[20,43],all_arms_reported=True,test_used_for_selection=False,
        source_hashes={f:sha256(dst/f) for f in ("views.npz","render_audit.json","build_views.py","train.py")},
    )
    models={}
    for seed in seeds:
        for arm in ("no_augmentation","repeat","two_suffix","multi_suffix"):
            if arm=="no_augmentation":
                entry=old_seal["models"][f"{seed}_state_host10_none_bce"]
                path=Path(entry["path"])
                assert sha256(path)==entry["sha256"]
                model=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
            else:
                addon=parent.copy()
                if arm=="two_suffix":
                    addon[:,:,:768]+=var_delta[:n,0:1,:]
                elif arm=="multi_suffix":
                    addon[:,:,:768]+=var_delta[:n,1:,:]
                x=np.vstack((tr,addon.reshape(-1,tr.shape[1])))
                model=lgb.train(cfg | {"seed":seed},lgb.Dataset(x,label=augmented,weight=weights))
                path=result_dir/f"{seed}_{arm}.txt"
                path.write_text(model.model_to_string(),encoding="utf-8")
            s=model.predict(va,num_threads=cfg["num_threads"])
            cuts=operating_points(valid["y"],s)
            if arm=="no_augmentation":
                for key,cut in cuts.items():
                    np.testing.assert_allclose(cut,entry["thresholds"][key],rtol=0,atol=2e-15)
            pair=model.predict(vparent.reshape(-1,va.shape[1]),num_threads=cfg["num_threads"]).reshape(len(valid_ix),-1)
            original_scores=s[valid_ix]
            probabilities=np.column_stack((original_scores,pair))
            spread=probabilities.max(axis=1)-probabilities.min(axis=1)
            paired=dict(n=len(pair),labeled_positives=int(valid["y"][valid_ix].sum()),
                        mean_multisuffix_probability_range=float(spread.mean()),
                        median_multisuffix_probability_range=float(np.median(spread)),
                        p90_multisuffix_probability_range=float(np.quantile(spread,.9)),
                        changed_class={key:int((np.any((probabilities>=cut)!=(original_scores[:,None]>=cut),axis=1)).sum())
                                       for key,cut in cuts.items()})
            key=f"{seed}_{arm}"
            models[key]=dict(arm=arm,seed=seed,path=str(path),sha256=sha256(path),cuts=cuts,paired_validation=paired,
                             validation={k:result_at(valid["y"],s,t) for k,t in cuts.items()})
            np.savez_compressed(result_dir/f"{key}_validation.npz",ids=valid["ids"],y=valid["y"],scores=s,
                                paired_ids=valid["ids"][valid_ix],paired_scores=pair)
            print("validation",key,round(models[key]["validation"]["max_f1"]["ap"],5),flush=True)
    seal=result_dir/"validation_seal.json"
    seal.write_text(json.dumps(dict(protocol=protocol,models=models),indent=2),encoding="utf-8")
    assert sha256(source/"test.npz")==receipt["test_sha256"]
    test=load_role(source,"test.npz","test",None)
    x=append(test["state"],test["history"],"host10")
    evidence=json.loads((base/"AIT反事实错例审计_20261105_v1/source_evidence.json").read_text(encoding="utf-8"))
    positives=test["ids"][test["y"]==1]
    assert all(s in evidence and evidence[s]["official_tags"] for s in positives)
    dns=np.array([s for s in positives if "dnsteal" in evidence[s]["official_tags"]])
    difficult=("wilson_000037","wilson_000376","wilson_000382","wilson_000466")
    assert set(difficult)<=set(dns)
    for key,entry in models.items():
        path=Path(entry["path"])
        assert sha256(path)==entry["sha256"]
        m=lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        s=m.predict(x,num_threads=cfg["num_threads"])
        order=np.lexsort((test["ids"],-s))
        ranks=np.argsort(order)+1
        entry["test"]={name:result_at(test["y"],s,t) for name,t in entry["cuts"].items()}
        entry["test_top_k"]=top_budgets(test["y"],test["ids"],s)
        dnsmask=np.isin(test["ids"],dns)
        entry["dns"]={name:int((dnsmask&(s>=t)).sum()) for name,t in entry["cuts"].items()}
        entry["dns_top43"]=int(dnsmask[order[:43]].sum())
        entry["known_difficult_cases"]={sid:dict(rank=int(ranks[i]),score=float(s[i]))
                                       for i,sid in enumerate(test["ids"]) if sid in difficult}
        entry["test_by_scene"]={str(scene):{name:result_at(test["y"][mask],s[mask],t)
                                           for name,t in entry["cuts"].items()}
                                for scene in np.unique(test["scene"]) if (mask:=test["scene"]==scene).any()}
        np.savez_compressed(result_dir/f"{key}_test.npz",ids=test["ids"],y=test["y"],scores=s)
    result=dict(protocol=protocol,validation_seal_sha256=sha256(seal),dns_test_size=len(dns),models=models)
    (result_dir/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("complete",len(models),"models",flush=True)

if __name__=="__main__":
    main()


