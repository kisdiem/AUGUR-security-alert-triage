"""Recompute the common AIT model tables from sealed history readouts."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
DESIGN = BASE / "ait_422_entity_redesign_20261115_v1"
spec = spec_from_file_location("direct_reference", DESIGN / "run.py")
direct = module_from_spec(spec)
spec.loader.exec_module(direct)
c, f = direct.c, direct.frozen
EXP = direct.PRIOR
OLD = BASE / "ait_augur_action_contrast_20261115_v1/output"
OLD_INDEX = c.read(OLD / "complete.json")
REDESIGN = DESIGN / "output"
REDESIGN_INDEX = c.read(REDESIGN / "complete.json")
FEATURES = DESIGN / "features"
OUT = ROOT / "output"
JOBS = [(tag, seed, f"fold_{i:02d}") for i in range(1, 9) for tag in ("611", "422") for seed in c.CONFIG["seeds"]]
PINS = {k: c.sha(v) for k,v in {
    "plan": ROOT/"plan.json", "run": ROOT/"run.py", "history": FEATURES/"complete.json",
    "expression": EXP/"output/complete.json", "old": OLD/"complete.json",
    "redesign": REDESIGN/"complete.json", "redesign_code": DESIGN/"run.py",
}.items()}


def role_inputs(x, ids):
    all_rows, _ = direct.get_design(x, ids)
    h = direct.views(c.host_view(x, ids), x)["host_profile"]
    return {"current": x[:, :811], "joint": all_rows["joint_roles_links"],
            "entity": all_rows["entity_roles_links"], "host": h}


def read_scores(directory, index, prefix, role, ids, y):
    name = prefix + f".{role}.npz"
    with np.load(c.pin(directory/name, index["files"][name]), allow_pickle=False) as source:
        np.testing.assert_array_equal(source["ids"], ids)
        np.testing.assert_array_equal(source["y"], y)
        return {k:source[k] for k in source.files if k not in ("ids", "y")}


def augment(tag, fold):
    x, ids, y, weights, vi, vy, vx, coverage = c.augmented_data(tag, fold)
    n = len(ids) - sum(coverage.get("counts", [coverage.get("parents", 0)]))
    if tag == "422":
        additions = [direct.alternate(x[:n], ids[:n], direct.pool(fold, fam)) for fam in direct.FAMILIES]
        x = np.r_[x, *[rows for _, rows in additions]]
        y = np.r_[y, *[y[idx] for idx, _ in additions]]
        ids = np.r_[ids, *[ids[idx] for idx, _ in additions]]
    natural = {sid:i for i,sid in enumerate(ids[:n])}
    assert len(natural) == n and not set(ids) & set(vi)
    parents = np.asarray([natural[sid] for sid in ids[n:]], dtype=int)
    np.testing.assert_array_equal(y[parents], y[n:])
    return x, np.r_[x[:n], x[parents]], ids, y, weights, vi, vy, vx, n


def fit(job):
    tag, seed, fold = job
    prefix = f"{tag}_{fold}_{seed}"
    decision = OUT/(prefix + ".seal.json")
    artifact = OUT/(prefix + ".joblib")
    validation = OUT/(prefix + ".validation.npz")
    if decision.is_file():
        s = c.read(decision)
        assert s["pins"] == PINS
        c.pin(artifact, s["model"])
        c.pin(validation, s["validation"])
        return prefix
    aug, rep, ids, y, w, vi, vy, vx, n = augment(tag, fold)
    sources = {"0": role_inputs(rep, ids), "1": role_inputs(aug, ids)}
    local, *_ = c.f.local_views(ids[:n], y[:n])
    model = {}
    # All combinations; the local scorer is applied after the main tree head.
    rules = {"00": ("current", "0"), "01": ("current", "1"), "10": ("joint", "0"),
             "11": ("joint", "1"), "entity_only_profile": ("entity","1"), "host_profile": ("host","1")}
    for arm, (view, treatment) in rules.items():
        data = sources[treatment][view]
        head = c.f.base_fit(data, y, seed, width=data.shape[1], weights=w)
        context = None
        if tag == "611":
            cx = np.c_[data[:n], local[:, 20:]]
            context = c.f.base_fit(cx, y[:n], seed, width=cx.shape[1], trees=160)
        model[arm] = (view, head, context)
    local_heads = c.models(job)[-1]
    score, alpha = predict(job, "validation", vi, vy, vx, local_heads, model, None)
    c.npz(validation, ids=vi, y=vy, **score)
    tmp = artifact.with_suffix(".partial")
    c.joblib.dump(model, tmp, compress=1)
    tmp.replace(artifact)
    c.save(decision, {"pins": PINS, "model": c.sha(artifact), "validation": c.sha(validation),
                      "alpha": alpha, "train_sizes": [n,len(y)],
                      "cutoff": {key:f.cutoff(vy,p) for key,p in score.items()}})
    print("sealed", prefix, flush=True)
    return prefix


def predict(job, role, ids, y, x, local_heads, model, saved):
    tag, seed, fold = job
    prefix = f"{tag}_{fold}_{seed}"
    arrays = role_inputs(x,ids)
    local, gates, *_ = c.f.local_views(ids,y)
    q = c.f.local_score(local_heads,local,gates)
    score, alpha = {}, {}
    for a,(view,head,context) in model.items():
        data = arrays[view]
        primary = head.predict_proba(data)[:,1]
        base = f.continuous(primary,q)
        residual = f.context_delta(base,data,local,context)
        alpha[a] = saved[a] if saved is not None else c.promotion(y,base,residual) if context is not None else 0.
        if len(a) == 2:
            score[a+"0"] = primary
            score[a+"1"] = base+alpha[a]*residual
        else:
            score[a] = base+alpha[a]*residual
    if tag == "422":
        prior = read_scores(REDESIGN, REDESIGN_INDEX, prefix, role, ids, y)
        for current,old in (("111","joint_roles_links"),("host_profile","host"),("entity_only_profile","entity_roles_links")):
            np.testing.assert_array_equal(score[current],prior[old])
        score["previous_augur"] = prior["old_joint"]
    else:
        prior = read_scores(OLD, OLD_INDEX, prefix, role, ids, y)
        np.testing.assert_array_equal(score["host_profile"],prior["host_profile"])
        for off in ("000","001","010","011"):
            np.testing.assert_array_equal(score[off],prior[off])
        score["previous_augur"] = prior["111"]
    key = c.tag_key(job)
    anchor = c.paths(job, f.OLD.parent/"ait_augur_host_entity_20261114_v1/paper_revision_20261114")
    index = c.read(anchor.parents[3]/"complete.json")
    seal = c.read(c.pin(anchor/"validation_seal.json",index["validation_seals"][key]))
    fname = anchor/f"{role}.npz"
    with np.load(c.pin(fname,seal["validation"] if role=="validation" else index["test_files"][key]),allow_pickle=False) as z:
        np.testing.assert_array_equal(ids,z["ids"])
        np.testing.assert_array_equal(y,z["y"])
        score["current_only"] = z["current_only"]
    return score,alpha


def test(job):
    tag, seed, fold = job
    prefix = f"{tag}_{fold}_{seed}"
    path = OUT/(prefix + ".test.npz")
    done = OUT/(prefix + ".done.json")
    lock = OUT/"validation_complete.json"
    seal = c.read(c.pin(OUT/(prefix+".seal.json"),c.read(lock)["validation"][prefix]))
    if done.is_file():
        status = c.read(done)
        assert status["lock"] == c.sha(lock)
        c.pin(path,status["test"])
        return
    ids,y,x = f.verified_roles(job)
    _,tr,_,_,vi,*_ = c.augmented_data(tag,fold)
    assert not set(ids)&(set(tr)|set(vi))
    model = c.joblib.load(c.pin(OUT/(prefix+".joblib"),seal["model"]))
    score,_ = predict(job,"test",ids,y,x,c.models(job)[-1],model,seal["alpha"])
    c.npz(path,ids=ids,y=y,**score)
    c.save(done,{"lock":c.sha(lock),"test":c.sha(path)})
    print("tested",prefix,flush=True)


def main():
    direct.BANK={}
    for scene,fingerprint in c.read(FEATURES/"complete.json").items():
        folder = FEATURES/scene
        info = c.read(c.pin(folder/"complete.json",fingerprint))
        with np.load(c.pin(folder/"readout.npz",info["feature"]),allow_pickle=False) as z:
            direct.BANK.update(zip(z["ids"],zip(z["roles"],z["links"],strict=True),strict=True))
    assert len(direct.BANK)==11482
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AUGUR_CPU_WORKERS","8"))) as pool:
        names = list(pool.map(fit,JOBS))
        c.save(OUT/"validation_complete.json",{"pins":PINS,"validation":{n:c.sha(OUT/(n+".seal.json")) for n in names}})
        list(pool.map(test,JOBS))
    files=[OUT/(n+s) for n in names for s in (".test.npz",".validation.npz",".seal.json",".done.json")]
    files.append(OUT/"validation_complete.json")
    receipt=OUT/"complete.json"
    c.save(receipt,{"pins":PINS,"files":{p.name:c.sha(p) for p in files},"cells":len(names)})
    temp=ROOT/"output.partial"
    with tarfile.open(temp,"w:gz") as bundle:
        for p in [*files,receipt]:
            bundle.add(p,arcname=p.name)
    temp.replace(ROOT/"output.tar.gz")


if __name__=="__main__":
    main()

