"""Verify selected AIT AUGUR predictions and generate the manuscript's affected results."""
from collections import defaultdict
import csv
import json
from hashlib import sha256
from itertools import product
from pathlib import Path
from statistics import stdev
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "ait_full_same_fusion_20261115_v1"
FIRST = ROOT.parent / "ait_history_complement_20261115_v1" / "collected"
SECOND = ROOT.parent / "ait_history_complement_20261115_v2" / "collected"
P = ROOT / "collected"
SOURCE = ROOT.parent / "ait_augur_host_entity_20261114_v1"
REPLAY = SOURCE / "paper_revision_20261114" / "replay_scores"
COST_DIR = ROOT.parent / "ait_422_error_audit_20261113_v1"
sys.path[:0] = [str(OLD), str(ROOT.parent / "ait_record_budget_v1"),
                str(SOURCE / "paper_revision_20261114")]
from verify import read, digest, point, zip_load, SEEDS, TAGS
from run import select as record_budget_select
from route_missing_history import route


FLAGS = tuple("".join(map(str, t)) for t in product(range(2), repeat=3))
FALLBACKS = ("host_profile", "entity_only_profile")
ARMS = (*FLAGS, *FALLBACKS, "current_only", "previous_augur")
FRACTIONS = (0.1, 0.3, 0.5, 1.0)
PAIRS = (("111","current_only"), ("111","previous_augur"), ("111","host_profile"), ("111","entity_only_profile"),
         ("111","011"), ("111","101"), ("111","110"), ("independent_0.3","111"),
         ("common_0.3","111"), ("common_1","111"))


def write(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def threshold(y, p):
    n = np.sort(p[y == 0])[::-1]
    return float(np.nextafter(n[len(n)//100], np.inf))


def load_costs():
    with (COST_DIR / "by_session.csv").open(newline="", encoding="utf-8-sig") as f:
        result = {r["id"]:(int(r["y"]),int(r["alerts"])) for r in csv.DictReader(f)}
    assert len(result) == 11482 and sum(y for y,_ in result.values()) == 403
    return result


def outage(ids, arm, rate):
    return np.asarray([int.from_bytes(sha256(f"history-v1/{arm}/{sid}".encode()).digest()[:8],"big") / 2**64 < rate
                       for sid in ids])


def read_cell(folder, index, name):
    return zip_load(folder / name, index["files"][name])


def match(left, right, a, b):
    for k in ("ids", "y"):
        np.testing.assert_array_equal(left[k], right[k])
    np.testing.assert_array_equal(left[a], right[b])


def frozen_summary(rows, count_rows, record_rows):
    summaries = {}
    draws = np.random.default_rng(20261107).integers(0,8,(10000,8))
    for tag in TAGS:
        scenes = sorted({s for t,_,s,_ in rows if t==tag})
        names = sorted({a for t,a,_,_ in rows if t==tag})
        assert len(scenes) == 8
        n = 1 if tag == "611" else 2
        aggregates, deviations = {}, {}
        for name in names:
            assert all(len(rows[tag,name,s,seed]) == n for s in scenes for seed in SEEDS)
            grid = np.asarray([[np.mean(rows[tag,name,s,seed], axis=0) for seed in SEEDS] for s in scenes])
            aggregates[name] = grid.mean(1)
            deviations[name] = stdev(grid[:, :, 0].mean(0))
        pairs = {}
        for a,b in PAIRS:
            diff = 100*(aggregates[a]-aggregates[b])
            pairs[a+"_vs_"+b] = {"mean_pp":diff.mean(0).tolist(),
                                "paired95_pp":np.quantile(diff[draws].mean(1),[.025,.975], axis=0).T.tolist(),
                                "wins":(diff > 1e-10).sum(0).tolist(), "losses":(diff < -1e-10).sum(0).tolist()}
        summaries[tag] = {
            "means":{a:m.mean(0).tolist() for a,m in aggregates.items()},
            "ap_seed_sd":deviations, "pairs":pairs,
            "scenes":{a:dict(zip(scenes,m.tolist(),strict=True)) for a,m in aggregates.items()},
            "record_budget":{a:np.sum(record_rows[tag,a],axis=0).tolist() for a in ("111","current_only","previous_augur")},
            "outage_fraction":{a:(sum(x for x,_ in count_rows[tag,a])/sum(n for _,n in count_rows[tag,a]))
                               for t,a in count_rows if t==tag},
        }
    return summaries


def main():
    receipt, main_ref, host_ref, old_ref, replay = [
        read(p / "complete.json") for p in (P,SECOND,FIRST,OLD/"collected",REPLAY)]
    assert receipt["cells"] == main_ref["fold_seeds"] == host_ref["fold_seeds"] == 80
    assert len(receipt["files"]) == 320
    for key,name in (("plan","plan.json"),("code","fit.py"),("features","history_features.py")):
        assert receipt[key] == digest(ROOT / name)
    assert receipt["source"] == {"host_profile":digest(FIRST/"complete.json"),"profile_contrast":digest(SECOND/"complete.json")}
    hashes = read(SOURCE / "scorer_lock.json")
    costs = load_costs()
    rows, counts, record = defaultdict(list), defaultdict(list), defaultdict(list)
    detail = []
    locks = {}
    for tag, seed, fold_num in product(TAGS, SEEDS, range(1,9)):
        fold = f"fold_{fold_num:02d}"
        pre = f"{tag}_{fold}_{seed}"
        key = f"{tag}/{seed}/{fold}"
        done = read(P/(pre+".json"))
        seal = read(P/(pre+".seal.json"))
        for suf in (".json",".seal.json"):
            assert digest(P/(pre+suf)) == receipt["files"][pre+suf]
        assert (done["pins"] == seal["pins"] ==
                {k:receipt[k] for k in ("plan","code","features","source")})
        assert done["model"] == seal["model"]
        assert done["validation"] == seal["validation"] == receipt["files"][pre+".validation.npz"]
        assert done["test"] == receipt["files"][pre+".test.npz"]
        val = read_cell(P,receipt,pre+".validation.npz")
        test = read_cell(P,receipt,pre+".test.npz")
        assert not set(val["ids"])&set(test["ids"])
        for name, arr in (("validation",val),("test",test)):
            target = pre + f".{name}.npz"
            match(arr, read_cell(SECOND,main_ref,target), "111","profile_contrast")
            match(arr, read_cell(FIRST,host_ref,target), "host_profile","host_profile")
        assert seal["cutoff"] == {a:threshold(val["y"],val[a]) for a in (*FLAGS,*FALLBACKS)}
        assert set(seal["alpha"]) == {"00","01","10","11",*FALLBACKS}
        assert all(a in ((0.,.25,.5,1.) if tag=="611" else (0.,)) for a in seal["alpha"].values())
        main_seal = SECOND / (pre+".seal.json")
        host_seal = FIRST / (pre+".seal.json")
        assert digest(main_seal) == main_ref["files"][pre+".seal.json"]
        assert digest(host_seal) == host_ref["files"][pre+".seal.json"]
        ms, hs = read(main_seal), read(host_seal)
        assert seal["alpha"]["11"] == ms["alpha"]["profile_contrast"]
        assert seal["alpha"]["host_profile"] == hs["alpha"]["host_profile"]
        locks[key] = {"main_seal":digest(main_seal), "main_model":ms["model"],
                      "host_seal":digest(host_seal), "host_model":hs["model"],
                      "control_seal":done["seal"], "control_model":done["model"],
                      "local_seal":hashes[key]["entity_seal"]}
        old = REPLAY / "results" / tag / f"seed_{seed}" / fold
        old_seal = read(old/"validation_seal.json")
        assert digest(old/"validation_seal.json") == replay["validation_seals"][key]
        for name, arr in (("validation",val),("test",test)):
            orig = zip_load(old/(name+".npz"),old_seal["validation"] if name == "validation" else replay["test_files"][key])
            past = read_cell(OLD/"collected",old_ref,pre+f".{name}.npz")
            for array in (orig,past):
                for k in ("ids","y"):
                    np.testing.assert_array_equal(arr[k],array[k])
            arr["current_only"] = orig["current_only"]
            arr["previous_augur"] = past["joint_continuous"]
        ids,y = test["ids"],test["y"]
        labels, record_cost = np.array([costs[i] for i in ids]).T
        np.testing.assert_array_equal(labels,y)
        assert np.all(record_cost>0)
        cut = {a:threshold(val["y"],val[a]) for a in ARMS}
        values = {a:test[a] for a in ARMS}
        for mode,rate in product(("independent","common"),FRACTIONS):
            h = outage(ids,"common" if mode=="common" else "host",rate)
            e = outage(ids,"common" if mode=="common" else "entity",rate)
            name = f"{mode}_{rate:g}"
            values[name] = route(test["host_profile"],test["entity_only_profile"],test["011"],test["111"],~h,~e)
            cut[name] = cut["111"]
            counts[tag,name].append(((h & e).sum(),len(y)))
        np.testing.assert_array_equal(values["common_1"],values["011"])
        np.testing.assert_array_equal(values["independent_1"],values["011"])
        for arm in ("current_only","111","previous_augur"):
            order = np.lexsort((ids,-values[arm]))
            budget = int(record_cost.sum())*2//100
            chosen, used = record_budget_select(order,record_cost,budget,False)
            assert used <= budget
            record[tag,arm].append((int(y[chosen].sum()),len(chosen),used,budget))
        scene_ids = np.array([sid.split("_",1)[0] for sid in ids])
        for s in np.unique(scene_ids):
            ok = scene_ids == s
            truth = y[ok]
            for name,p in values.items():
                score = p[ok]
                ap, recall = point(truth,score)
                sent = score >= cut[name]
                result = (ap, recall, sent[truth==1].mean(), sent[truth==0].mean())
                rows[tag,name,str(s),seed].append(result)
                detail.append((tag,fold,seed,str(s),name,*result,cut[name]))
    results = frozen_summary(rows,counts,record)
    ref_metrics = read(SECOND.parent/"results.json")["data"]
    last = read(OLD/"paper_metrics.json")["results"]
    for tag in TAGS:
        means = results[tag]["means"]
        for new, old in (("111","profile_contrast"),("host_profile","host_profile"),("previous_augur","joint_continuous")):
            np.testing.assert_allclose(means[new],ref_metrics[tag]["means"][old],atol=1e-13,rtol=0.)
        np.testing.assert_allclose(means["current_only"][:2],last[tag]["means"][0],atol=1e-13,rtol=0.)
    membership = {"source":digest(P/"complete.json"),"history_features_sha256":digest(ROOT/"history_features.py"),
                  "fit_sha256":digest(ROOT/"fit.py"), "score_arm":"profile_contrast", "models":locks}
    write(ROOT/"selected_model.json",membership)
    write(ROOT/"paper_results.json",{"metrics":["AP","Recall@1% FPR","frozen_recall","frozen_negative_send"],
                                     "source":membership["source"],"cells":80,"record_budget_columns":
                                     ["positive_hits","selected_sessions","records_used","budget"],"results":results})
    with (ROOT/"per_scene.csv").open("w",newline="",encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(("protocol","fold","seed","scene","arm","AP","Recall@1% FPR","frozen_recall","frozen_negative_send","validation_cut"))
        w.writerows(detail)
    for tag in TAGS:
        print(tag)
        for arm in ARMS+("independent_0.3","common_0.3","common_1"):
            print(arm, *(round(p*100,4) for p in results[tag]["means"][arm]))


if __name__ == "__main__":
    main()

