"""Compare redesigned and frozen entity readouts under the same AIT protocol."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "ait_422_expression_shift_20261115_v1"
DONORS = ROOT.parent / "ait_entity_evidence_fusion_20261114_v1/bundle"
sys.path.insert(0, str(PRIOR))
from fit import EDITS, FAMILIES, TRANSFORMS, alternate, c, frozen
from history_features import views

OUT = ROOT / "output"
FEATURES = ROOT / "features"
INDEX = c.read(PRIOR / "output/complete.json")
EDIT_INDEX = c.read(c.pin(EDITS / "complete.json", INDEX["edits"]))
BASELINES = dict(zip(("host", "old_entity", "old_joint"), TRANSFORMS, strict=True))
NEW = ("joint_no_entity_volume", "joint_roles", "joint_links", "joint_roles_links", "entity_roles_links")
PROBES = ("new_same_host_other_history", "old_swap_volume_age", "old_swap_actions")
JOBS = [("422", seed, f"fold_{i:02d}") for i in range(1, 9) for seed in c.CONFIG["seeds"]]


def get_prior(prefix, role, ids, y):
    file = prefix + f".{role}.npz"
    with np.load(c.pin(PRIOR / "output" / file, INDEX["files"][file]), allow_pickle=False) as old:
        np.testing.assert_array_equal(ids, old["ids"])
        np.testing.assert_array_equal(y, old["y"])
        return {key: old["edit/" + arm] for key, arm in BASELINES.items()}


def pool(fold, family):
    name = fold + "_" + family + ".npz"
    with np.load(c.pin(EDITS / name, EDIT_INDEX["files"][name]), allow_pickle=False) as f:
        return dict(zip(f["ids"], zip(f["original"], f["alternate"], strict=True), strict=True))


def get_design(x, ids):
    v = views(c.host_view(x, ids), x)
    h, j = v["host_profile"], v["profile_contrast"]
    roles = np.stack([BANK[s][0] for s in ids])
    links = np.stack([BANK[s][1] for s in ids])
    return {
        "joint_no_entity_volume": np.c_[h, j[:, 951:]],
        "joint_roles": np.c_[h, roles],
        "joint_links": np.c_[h, links],
        "joint_roles_links": np.c_[h, roles, links],
        "entity_roles_links": np.c_[x[:, :811], roles, links],
    }, j


def local_score(job, ids, y):
    *_, heads = c.models(job)
    local, gate, *_ = c.f.local_views(ids, y)
    return c.f.local_score(heads, local, gate)


def score_new(models, rows, gate):
    return {name: frozen.continuous(models[name].predict_proba(rows[name])[:, 1], gate) for name in NEW}


def fit(job):
    tag, seed, fold = job
    prefix = f"{tag}_{fold}_{seed}"
    model_path = OUT / (prefix + ".joblib")
    validation = OUT / (prefix + ".validation.npz")
    decision = OUT / (prefix + ".decision.json")
    if decision.is_file():
        prev = c.read(decision)
        assert prev["pins"] == PINS
        c.pin(model_path, prev["model"])
        c.pin(validation, prev["validation"])
        return prefix
    x, ids, y, weights, vi, vy, vx, exposure = c.augmented_data(tag, fold)
    assert weights is None
    natural = len(ids) - exposure["parents"]
    name = prefix + ".seal.json"
    previous = c.read(c.pin(PRIOR / "output" / name, INDEX["files"][name]))
    copies = []
    for family in FAMILIES:
        p, xx = alternate(x[:natural], ids[:natural], pool(fold, family))
        assert previous["parents"][family] == {"total": len(p), "positive": int(y[p].sum())}
        copies.append((p, xx))
    train_ids = np.r_[ids, *[ids[ix] for ix, _ in copies]]
    yy = np.r_[y, *[y[ix] for ix, _ in copies]]
    assert not set(train_ids) & set(vi)
    inputs, _ = get_design(np.r_[x, *[xx for _, xx in copies]], train_ids)
    models = {name: c.f.base_fit(inputs[name], yy, seed, width=inputs[name].shape[1]) for name in NEW}
    rows, _ = get_design(vx, vi)
    p = {**get_prior(prefix, "validation", vi, vy),
         **score_new(models, rows, local_score(job, vi, vy))}
    c.npz(validation, ids=vi, y=vy, **p)
    tmp = model_path.with_suffix(".partial")
    c.joblib.dump(models, tmp, compress=1)
    tmp.replace(model_path)
    c.save(decision, {"pins": PINS, "model": c.sha(model_path), "validation": c.sha(validation),
                      "thresholds": {k: frozen.cutoff(vy, v) for k, v in p.items()},
                      "widths": {k: v.shape[1] for k, v in inputs.items()}, "parents": previous["parents"],
                      "train_rows": len(yy), "natural_train_rows": natural})
    print("sealed", prefix, flush=True)
    return prefix


def donor_indices(prefix, ids):
    name = prefix + ".npz"
    receipt = c.read(DONORS / "complete.json")
    with np.load(c.pin(DONORS / name, receipt["files"][name]), allow_pickle=False) as z:
        donor = dict(zip(z["ids"], z["donor_ids"][:, 0], strict=True))
    assert set(donor) == set(ids)
    index = {sid: i for i, sid in enumerate(ids)}
    mapped = np.asarray([index[donor[sid]] for sid in ids], dtype=int)
    np.testing.assert_array_equal(c.scene_codes(ids), c.scene_codes(ids[mapped]))
    return mapped


def diagnostic(prefix, ids, rows, j, q, models, original):
    donor = donor_indices(prefix.rsplit("_", 1)[0], ids)
    replacement = rows["joint_roles_links"].copy()
    replacement[:, 935:] = replacement[donor, 935:]
    alt = frozen.continuous(models["joint_roles_links"].predict_proba(replacement)[:, 1], q)
    seal = c.read(c.pin(PRIOR / "output" / (prefix + ".seal.json"), INDEX["files"][prefix + ".seal.json"]))
    old = c.joblib.load(c.pin(PRIOR / "output" / (prefix + ".joblib"), seal["model"]))["edit/profile_contrast"]
    np.testing.assert_array_equal(original, frozen.continuous(old.predict_proba(j)[:, 1], q))
    volume, actions = j.copy(), j.copy()
    volume[:, 935:951] = j[donor, 935:951]
    # Keep each target's host reference when replacing the donor's relative actions.
    actions[:, 951:] = (j[donor, 951:] + j[donor, 827:935]) - j[:, 827:935]
    # Reconstruct only for matched cases so a self fallback is exactly the original.
    actions[donor == np.arange(len(ids))] = j[donor == np.arange(len(ids))]
    pv = frozen.continuous(old.predict_proba(volume)[:, 1], q)
    pa = frozen.continuous(old.predict_proba(actions)[:, 1], q)
    return {"new_same_host_other_history": alt, "old_swap_volume_age": pv, "old_swap_actions": pa,
            "matched": donor != np.arange(len(ids))}


def test(job):
    tag, seed, fold = job
    prefix = f"{tag}_{fold}_{seed}"
    lock = OUT / "validation_complete.json"
    receipt = c.read(lock)
    decision = c.read(c.pin(OUT / (prefix + ".decision.json"), receipt["decisions"][prefix]))
    done = OUT / (prefix + ".done.json")
    path = OUT / (prefix + ".test.npz")
    if done.is_file():
        old = c.read(done)
        assert old["validation_lock"] == c.sha(lock)
        c.pin(path, old["test"])
        return
    ids, y, x = frozen.verified_roles(job)
    _, train, _, _, vi, _, _, _ = c.augmented_data(tag, fold)
    assert not set(ids) & (set(train) | set(vi))
    models = c.joblib.load(c.pin(OUT / (prefix + ".joblib"), decision["model"]))
    rows, j = get_design(x, ids)
    q = local_score(job, ids, y)
    baseline = get_prior(prefix, "test", ids, y)
    output = {**baseline, **score_new(models, rows, q),
              **diagnostic(prefix, ids, rows, j, q, models, baseline["old_joint"])}
    c.npz(path, ids=ids, y=y, **output)
    c.save(done, {"test": c.sha(path), "validation_lock": c.sha(lock)})
    print("tested", prefix, flush=True)


def main():
    global PINS, BANK
    replay_lock = FEATURES / "complete.json"
    while not replay_lock.is_file():
        print("waiting for replay receipt", flush=True)
        time.sleep(5)
    PINS = {key: c.sha(path) for key, path in {
        "plan": ROOT / "plan.json", "run": ROOT / "run.py", "replay": ROOT / "replay.py",
        "entity_state": ROOT / "entity_state.py", "baseline": PRIOR / "output/complete.json",
        "features": replay_lock, "donors": DONORS / "complete.json"}.items()}
    BANK = {}
    scene_info = []
    for scene, fingerprint in c.read(replay_lock).items():
        path = c.pin(FEATURES / scene / "complete.json", fingerprint)
        info = c.read(path)
        scene_info.append(path)
        with np.load(c.pin(FEATURES / scene / "readout.npz", info["feature"]), allow_pickle=False) as z:
            BANK.update(zip(z["ids"], zip(z["roles"], z["links"], strict=True), strict=True))
    assert len(BANK) == 11482
    OUT.mkdir(exist_ok=True)
    c.save(OUT / "features.json", {p.parent.name: c.read(p) for p in scene_info})
    with ProcessPoolExecutor(max_workers=int(os.environ.get("AUGUR_CPU_WORKERS", "8"))) as worker:
        prefixes = list(worker.map(fit, JOBS))
        c.save(OUT / "validation_complete.json", {"pins": PINS,
               "decisions": {n: c.sha(OUT / (n + ".decision.json")) for n in prefixes}})
        list(worker.map(test, JOBS))
    files = [OUT / (p + ext) for p in prefixes
             for ext in (".decision.json", ".validation.npz", ".done.json", ".test.npz")]
    files.extend((OUT / "validation_complete.json", OUT / "features.json"))
    complete = OUT / "complete.json"
    c.save(complete, {"pins": PINS, "files": {p.name: c.sha(p) for p in files}, "cells": len(prefixes)})
    staged = ROOT / "output.partial"
    with tarfile.open(staged, "w:gz") as pack:
        for path in [*files, complete]:
            pack.add(path, arcname=path.name)
    staged.replace(ROOT / "output.tar.gz")


if __name__ == "__main__":
    main()

