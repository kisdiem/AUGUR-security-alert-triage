"""Replay verified predictions and compare paired scene-level results."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

R=Path(__file__).resolve().parent
BASE=R.parent/"ait_augur_host_entity_20261114_v1"
P=R/"collected"
SEEDS=(42,1337,31415,2025,8675309)
TAGS=("611","422")
ARMS=("old_augur","host_continuous","entity_continuous","joint_continuous","weighted_continuous",
      "host_no_context","entity_no_context","joint_no_context","weighted_no_context")


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def zip_load(p,fingerprint):
    assert digest(p)==fingerprint,p
    with np.load(p,allow_pickle=False) as z:
        return dict(z)


def point(y,s):
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
    return average_precision_score(y,s), float(tpr[fpr<=.01].max())


def aggregate(rows,raw):
    out={}
    draw=np.random.default_rng(20261107).integers(0,8,(10000,8))
    for tag in TAGS:
        scenes=sorted({z for t,a,z in rows if t==tag})
        assert len(scenes)==8
        runs=5 if tag=="611" else 10
        assert all(len(rows[tag,a,s])==runs for a in ARMS for s in scenes)
        matrix={a:np.array([np.mean(rows[tag,a,s],axis=0) for s in scenes]) for a in ARMS}
        means={a:v.mean(0).tolist() for a,v in matrix.items()}
        ref=raw[tag]["means"]["validation_mix"]
        np.testing.assert_allclose(means["old_augur"][:2],[ref["AP"],ref["Recall@1% FPR"]],atol=1e-12,rtol=0)
        pairs={}
        for a in ("host_continuous","entity_continuous","weighted_continuous","joint_continuous","joint_no_context"):
            for comp in ("old_augur","host_continuous","entity_continuous","weighted_continuous"):
                delta=(matrix[a]-matrix[comp])*100
                pairs[a+"_vs_"+comp]={"delta_pp":delta.mean(0).tolist(),
                    "paired95_pp":np.quantile(delta[draw].mean(1),[.025,.975],axis=0).T.tolist(),
                    "ap_higher_scenes":int((delta[:,0]>1e-10).sum()),"ap_lower_scenes":int((delta[:,0]<-1e-10).sum())}
        out[tag]={"means":means,"comparisons":pairs,
                  "scenes":{a:dict(zip(scenes,v.tolist(),strict=True)) for a,v in matrix.items()}}
    return out


def main():
    receipt=read(P/"complete.json")
    ref=BASE/"paper_revision_20261114/replay_scores"
    original=read(ref/"complete.json")
    sources=read(BASE/"scorer_lock.json")
    assert receipt["cells"]==80 and len(receipt["files"])==320
    assert receipt["plan_sha256"]==digest(R/"plan.json") and receipt["code_sha256"]==digest(R/"run_full.py")
    assert receipt["old_prediction_index"]==digest(ref/"complete.json")
    assert all(digest(P/name)==h for name,h in receipt["files"].items())
    rows=defaultdict(list)
    for tag in TAGS:
        for index in range(1,9):
            fold=f"fold_{index:02d}"
            for seed in SEEDS:
                base=f"{tag}_{fold}_{seed}"
                old=ref/"results"/tag/f"seed_{seed}"/fold
                key=f"{tag}/{seed}/{fold}"
                o=read(old/"validation_seal.json")
                assert digest(old/"validation_seal.json")==original["validation_seals"][key]
                finished=read(P/(base+".json"))
                seal_path=P/(base+".seal.json")
                assert digest(seal_path)==finished["seal_sha256"]
                seal=read(seal_path)
                assert seal["model_sha256"]==finished["model_sha256"]
                assert seal["plan_sha256"]==receipt["plan_sha256"]
                assert seal["source_anchors"]=={"host":sources[key]["host_seal"],"entity":sources[key]["entity_seal"]}
                valid=zip_load(P/(base+".validation.npz"),seal["validation_sha256"])
                test=zip_load(P/(base+".test.npz"),finished["test_sha256"])
                for data,role in ((valid,"validation"),(test,"test")):
                    old_data=zip_load(old/(role+".npz"),o["validation"] if role=="validation" else original["test_files"][key])
                    for col in ("ids","y"):
                        np.testing.assert_array_equal(data[col],old_data[col])
                    w=sources[key]["weight"]
                    np.testing.assert_array_equal(data["old_augur"],(1-w)*old_data["host_cf_local"]+w*old_data["state_cf_local"])
                assert not set(valid["ids"]) & set(test["ids"])
                assert set(seal["cuts"])==set(ARMS)
                scene=np.array([s.split("_",1)[0] for s in test["ids"]])
                for arm in ARMS:
                    threshold=seal["cuts"][arm]
                    vn=valid["y"]==0
                    assert ((valid[arm]>=threshold)&vn).sum()<=vn.sum()//100
                    for s in np.unique(scene):
                        take=scene==s
                        y,p=test["y"][take],test[arm][take]
                        ap,recall=point(y,p)
                        accepted=p>=threshold
                        rows[tag,arm,str(s)].append((ap,recall,accepted[y==1].mean(),accepted[y==0].mean()))
    results=aggregate(rows,read(BASE/"fusion_300_reference.json")["protocols"])
    report={"metrics":["AP","Recall@1% FPR","frozen_threshold_recall","frozen_threshold_policy_negative_send_rate"],
            "run_sha256":digest(P/"complete.json"),"results":results}
    (R/"verified_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for tag,q in results.items():
        print(tag,{a:[round(x*100,4) for x in v] for a,v in q["means"].items()},flush=True)
        for pair in ("joint_continuous_vs_host_continuous","joint_continuous_vs_old_augur","weighted_continuous_vs_old_augur",
                     "joint_continuous_vs_weighted_continuous"):
            x=q["comparisons"][pair]
            print(pair,x,flush=True)


if __name__=="__main__":
    main()

