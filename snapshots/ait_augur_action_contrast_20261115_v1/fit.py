"""Recompute AUGUR ablation and read-failure controls for the selected action contrast."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "ait_full_same_fusion_20261115_v1"
FIRST = ROOT.parent / "ait_history_complement_20261115_v1" / "output"
SECOND = ROOT.parent / "ait_history_complement_20261115_v2" / "output"
sys.path.insert(0, str(OLD))
from run_full import c, continuous, context_delta, cutoff, verified_roles
from history_features import views

OUT = ROOT / "output"
REFS = {
    arm: (root, c.read(root / "complete.json"))
    for arm, root in (("host_profile", FIRST), ("profile_contrast", SECOND))
}
SOURCE = {arm: c.sha(root / "complete.json") for arm, (root, _) in REFS.items()}


def pinned(root, index, prefix, suffix):
    filename = prefix + suffix
    return c.pin(root / filename, index["files"][filename])


def load_reference(arm, prefix, exposure):
    root, index = REFS[arm]
    lock = c.read(pinned(root, index, prefix, ".seal.json"))
    assert exposure == lock["exposure"]
    pair = c.joblib.load(c.pin(root / (prefix + ".joblib"), lock["model"]))[arm]
    return pair, lock


def reference_scores(prefix, role, ids, y, scores):
    for arm, (root, index) in REFS.items():
        path = pinned(root, index, prefix, f".{role}.npz")
        with np.load(path, allow_pickle=False) as expected:
            np.testing.assert_array_equal(ids, expected["ids"])
            np.testing.assert_array_equal(y, expected["y"])
            name = "111" if arm == "profile_contrast" else arm
            np.testing.assert_array_equal(scores[name], expected[arm])


def fit_control(train, repeat, ids, labels, weights, exposure, seed, tag, reference):
    n = len(train) - sum(exposure.get("counts", [exposure.get("parents", 0)]))
    z, *_ = c.f.local_views(ids[:n], labels[:n])
    h = c.host_view(train, ids)
    f_cf = views(h, train)
    f_repeat = views(c.host_view(repeat, ids), repeat)
    current_train = train[:, :811]
    joint_train = f_cf["profile_contrast"]
    context = {}
    for name, x in (("0", current_train), ("entity_only_profile", f_cf["entity_only_profile"])):
        if tag == "611":
            ext = np.c_[x[:n], z[:, 20:]]
            context[name] = c.f.base_fit(ext, labels[:n], seed, width=ext.shape[1], trees=160)
        else:
            context[name] = None
    return {
        "00": (c.f.base_fit(repeat, labels, seed, width=811, weights=weights), context["0"]),
        "01": (c.f.base_fit(train, labels, seed, width=811, weights=weights), context["0"]),
        "10": (c.f.base_fit(f_repeat["profile_contrast"], labels, seed, width=joint_train.shape[1], weights=weights),
               reference["profile_contrast"][1]),
        "11": reference["profile_contrast"],
        "host_profile": reference["host_profile"],
        "entity_only_profile": (
            c.f.base_fit(f_cf["entity_only_profile"], labels, seed, width=935, weights=weights),
            context["entity_only_profile"],
        ),
    }


def make_scores(models, ids, y, x, local_heads, alpha=None):
    local, gates, *_ = c.f.local_views(ids, y)
    q = c.f.local_score(local_heads, local, gates)
    h = views(c.host_view(x, ids), x)
    matrices = {
        "00": x[:, :811], "01": x[:, :811],
        "10": h["profile_contrast"], "11": h["profile_contrast"],
        "host_profile": h["host_profile"], "entity_only_profile": h["entity_only_profile"],
    }
    scores = {}
    if alpha is None:
        alpha = {}
        calibrate = True
    else:
        calibrate = False
    for arm, (main, ctx) in models.items():
        raw = main.predict_proba(matrices[arm])[:, 1]
        base = continuous(raw, q)
        delta = context_delta(base, matrices[arm], local, ctx)
        if calibrate:
            alpha[arm] = c.promotion(y, base, delta) if ctx is not None else 0.
        if len(arm) == 2:
            scores[arm + "0"] = raw
            scores[arm + "1"] = base + alpha[arm] * delta
        else:
            scores[arm] = base + alpha[arm] * delta
    return scores, alpha


def run(job):
    tag, seed, fold = job
    prefix = f"{tag}_{fold}_{seed}"
    paths = {name: OUT / (prefix + suffix) for name, suffix in {
        "validation": ".validation.npz", "seal": ".seal.json",
        "test": ".test.npz", "done": ".json", "model": ".joblib"}.items()}
    pin = {"plan": c.sha(ROOT / "plan.json"), "code": c.sha(Path(__file__)),
           "features": c.sha(ROOT / "history_features.py"), "source": SOURCE}
    if paths["done"].exists():
        done = c.read(paths["done"])
        assert done["pins"] == pin
        for name in ("validation", "seal", "test", "model"):
            c.pin(paths[name], done[name])
        return prefix

    train, ids, y, weights, vi, vy, vx, exposure = c.augmented_data(tag, fold)
    _, es, _, hs, _, _, _, _, heads = c.models(job)
    assert exposure == es["augmentation"] == hs["exposure"]
    n = len(train) - sum(exposure.get("counts", [exposure.get("parents", 0)]))
    row = {sid: i for i, sid in enumerate(ids[:n])}
    parents = [row[sid] for sid in ids[n:]]
    np.testing.assert_array_equal(y[parents], y[n:])
    repeat = np.r_[train[:n], train[parents]]
    reference = {}
    alpha_expected = {}
    for arm in REFS:
        reference[arm], seal = load_reference(arm, prefix, exposure)
        alpha_expected["11" if arm == "profile_contrast" else arm] = seal["alpha"][arm]
    models = fit_control(train, repeat, ids, y, weights, exposure, seed, tag, reference)
    tmp = paths["model"].with_suffix(".tmp")
    c.joblib.dump({a: m for a, m in models.items() if a not in ("11","host_profile")}, tmp, compress=1)
    tmp.replace(paths["model"])

    val, alpha = make_scores(models, vi, vy, vx, heads)
    assert all(alpha[k] == a for k, a in alpha_expected.items())
    reference_scores(prefix, "validation", vi, vy, val)
    c.npz(paths["validation"], ids=vi, y=vy, **val)
    c.save(paths["seal"], {"pins": pin, "validation": c.sha(paths["validation"]),
                          "model": c.sha(paths["model"]), "alpha": alpha,
                          "cutoff": {a: cutoff(vy, p) for a, p in val.items()}})
    ti, ty, tx = verified_roles(job)
    assert not set(ti) & (set(ids) | set(vi))
    test, _ = make_scores(models, ti, ty, tx, heads, alpha)
    reference_scores(prefix, "test", ti, ty, test)
    c.npz(paths["test"], ids=ti, y=ty, **test)
    c.save(paths["done"], {"pins": pin, **{a: c.sha(paths[a]) for a in ("validation","test","seal","model")}})
    print("complete", prefix, flush=True)
    return prefix


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(p, seed, f"fold_{f:02d}") for f in range(1,9) for p in ("611","422") for seed in c.CONFIG["seeds"]]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AUGUR_CPU_WORKERS", "6"))) as pool:
        prefixes = list(pool.map(run, jobs))
    files = [OUT / (key + suffix) for key in prefixes
             for suffix in (".json",".seal.json",".validation.npz",".test.npz")]
    receipt = OUT / "complete.json"
    c.save(receipt, {"files": {p.name: c.sha(p) for p in files}, "cells":len(prefixes),
                     "plan":c.sha(ROOT/"plan.json"), "code":c.sha(Path(__file__)),
                     "features":c.sha(ROOT/"history_features.py"), "source":SOURCE})
    tmp = ROOT / "output.partial"
    with tarfile.open(tmp,"w:gz") as f:
        for p in [*files,receipt]:
            f.add(p,arcname=p.name)
    tmp.replace(ROOT/"output.tar.gz")


if __name__ == "__main__":
    main()

