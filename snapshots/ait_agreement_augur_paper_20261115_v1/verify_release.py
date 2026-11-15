"""Independently replay all Chapter 4 scores from the signed validation and test arrays."""
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path

from agreement import agreement
import csv

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / "ait_augur_action_contrast_20261115_v1"
DIRECT = BASE / "ait_422_entity_redesign_20261115_v1"
EXPR = BASE / "ait_422_expression_shift_20261115_v1"
ROLES = BASE / "ait_entity_roles_paper_20261115_v1"
EXPLORATORY = BASE / "ait_dual_protocol_stable_fusion_20261115_v2"
HYBRID = BASE / "ait_422_hybrid_judgment_20261115_v1"
P = ROOT / "collected"
spec = spec_from_file_location("sealed_ait_evaluator", OLD / "evaluate.py")
ref = module_from_spec(spec)
spec.loader.exec_module(ref)
ref.PAIRS += (("111","redesigned_joint"),)
ARMS = (*ref.ARMS,"redesigned_joint")


def main():
    receipt = ref.read(P/"complete.json")
    assert receipt["cells"] == 80 and len(receipt["files"]) == 321
    assert all(ref.digest(P/f) == h for f, h in receipt["files"].items())
    sources = {"plan":ROOT/"plan.json","run":ROOT/"run.py","agreement":ROOT/"agreement.py",
               "roles":ROLES/"collected/complete.json","legacy":OLD/"collected/complete.json",
               "edit":EXPR/"collected/complete.json","history":DIRECT/"readouts/complete.json"}
    assert receipt["pins"] == {k:ref.digest(v) for k,v in sources.items()}
    lock = ref.read(P/"validation_complete.json")
    assert lock["pins"] == receipt["pins"] and len(lock["validation"]) == 80
    role_index=ref.read(ROLES/"collected/complete.json")
    hybrid_index=ref.read(HYBRID/"collected/complete.json")
    features=DIRECT/"readouts"
    support={}
    for scene,h in ref.read(features/"complete.json").items():
        folder=features/scene
        assert ref.digest(folder/"complete.json")==h
        info=ref.read(folder/"complete.json")
        a=ref.zip_load(folder/"readout.npz",info["feature"])
        support.update(zip(a["ids"],zip(np.any(a["roles"]>0,axis=1),np.any(a["links"]>0,axis=1),strict=True),strict=True))
    assert len(support)==11482
    replay = BASE/"ait_augur_host_entity_20261114_v1/paper_revision_20261114/replay_scores"
    base = ref.read(replay/"complete.json")
    costs = ref.load_costs()
    rows, counts, budgets = defaultdict(list), defaultdict(list), defaultdict(list)
    detail = []
    for tag, seed, i in product(ref.TAGS, ref.SEEDS, range(1,9)):
        fold = f"fold_{i:02d}"
        prefix = f"{tag}_{fold}_{seed}"
        key = f"{tag}/{seed}/{fold}"
        s = ref.read(P/(prefix+".seal.json"))
        done = ref.read(P/(prefix+".done.json"))
        assert s["pins"] == receipt["pins"]
        assert lock["validation"][prefix] == receipt["files"][prefix+".seal.json"]
        assert done["lock"] == receipt["files"]["validation_complete.json"]
        assert s["validation"] == receipt["files"][prefix+".validation.npz"]
        assert done["test"] == receipt["files"][prefix+".test.npz"]
        val = ref.zip_load(P/(prefix+".validation.npz"),s["validation"])
        test = ref.zip_load(P/(prefix+".test.npz"),done["test"])
        assert not set(val["ids"]) & set(test["ids"])
        assert s["alpha"] in ((0.,.25,.5,1.) if tag=="611" else (0.,))
        assert s["cutoff"] == {a:ref.threshold(val["y"],p) for a,p in val.items() if a not in ("ids","y")}
        anchor = replay/"results"/tag/f"seed_{seed}"/fold
        assert ref.digest(anchor/"validation_seal.json") == base["validation_seals"][key]
        prev = ref.read(anchor/"validation_seal.json")
        for role, value in (("validation",val), ("test",test)):
            filename = prefix+f".{role}.npz"
            archived = ref.read_cell(ROLES/"collected",role_index,filename)
            reference = ref.zip_load(anchor/f"{role}.npz",prev["validation"] if role=="validation" else base["test_files"][key])
            ref.match(value,reference,"current_only","current_only")
            for a in ("000","001","010","011","host_profile","entity_only_profile","previous_augur"):
                ref.match(value,archived,a,a)
            ref.match(value,archived,"redesigned_joint","111")
            rp, rel = np.asarray([support[s] for s in value["ids"]],dtype=bool).T
            expected=agreement(archived["previous_augur"],archived["host_profile"],archived["111"],rp,rel)
            np.testing.assert_array_equal(value["111"],expected)
        y,ids = test["y"],test["ids"]
        values={k:test[k] for k in ARMS}
        cutoff=dict(s["cutoff"])
        if tag=="422":
            benchmark={}
            for role,z in (("validation",val),("test",test)):
                existing=ref.read_cell(HYBRID/"collected",hybrid_index,prefix+f".{role}.npz")
                ref.match(z,existing,"host_profile","forest/host_profile")
                benchmark[role]=existing["fixed_blend/host_profile"]
            values["host_hybrid"]=benchmark["test"]
            cutoff["host_hybrid"]=ref.threshold(val["y"],benchmark["validation"])
        for mode,rate in product(("independent","common"),ref.FRACTIONS):
            h = ref.outage(ids,"common" if mode=="common" else "host",rate)
            e = ref.outage(ids,"common" if mode=="common" else "entity",rate)
            name = f"{mode}_{rate:g}"
            values[name] = ref.route(test["host_profile"],test["entity_only_profile"],test["011"],test["111"],~h,~e)
            cutoff[name] = cutoff["111"]
            counts[tag,name].append(((h&e).sum(),len(y)))
        np.testing.assert_array_equal(values["common_1"],values["011"])
        matched_labels,record_cost=np.array([costs[k] for k in ids]).T
        np.testing.assert_array_equal(y,matched_labels)
        for arm in ("current_only","111","previous_augur"):
            order=np.lexsort((ids,-values[arm]))
            budget=int(record_cost.sum())*2//100
            picked,spent=ref.record_budget_select(order,record_cost,budget,False)
            assert spent<=budget
            budgets[tag,arm].append((int(y[picked].sum()),len(picked),spent,budget))
        scenes=np.array([sid.split("_",1)[0] for sid in ids])
        for a,p in values.items():
            for scene in np.unique(scenes):
                mask=scenes==scene
                truth,score=y[mask],p[mask]
                called=score>=cutoff[a]
                result=(*ref.point(truth,score),called[truth==1].mean(),called[truth==0].mean())
                rows[tag,a,str(scene),seed].append(result)
                detail.append((tag,fold,seed,str(scene),a,*result,cutoff[a]))
    results=ref.frozen_summary(rows,counts,budgets)
    earlier=ref.read(EXPLORATORY/"results.json")["results"]
    for t in ref.TAGS:
        np.testing.assert_allclose(results[t]["means"]["111"],earlier[t]["means"]["agreement"],rtol=0.,atol=1e-13)
    np.testing.assert_allclose(results["422"]["means"]["host_hybrid"],earlier["422"]["means"]["strong_host_neural"],
                               rtol=0.,atol=1e-13)
    ref.write(ROOT/"paper_results.json",{"metrics":["AP","Recall@1% FPR","frozen_recall","frozen_negative_send"],
             "source":ref.digest(P/"complete.json"),"cells":80,"results":results})
    with (ROOT/"per_scene.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("protocol","fold","seed","scene","arm","AP","Recall@1% FPR",
                         "frozen_recall","frozen_negative_send","validation_cut"))
        writer.writerows(detail)
    for tag in ref.TAGS:
        print(tag,{a:[round(x*100,4) for x in metrics[:2]] for a,metrics in results[tag]["means"].items()})


if __name__=="__main__":
    main()



