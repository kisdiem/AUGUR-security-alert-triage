"""Recompute held-out metrics and decisions from the released AIT score arrays."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROTATIONS = {"611": 1, "422": 2}
SEEDS = (42, 1337, 31415, 2025, 8675309)
CELLS = {
    f"{rotation}_fold_{fold:02d}_{seed}": (rotation, fold, seed)
    for rotation, fold, seed in product(ROTATIONS, range(1, 9), SEEDS)
}
ARMS = {
    "original_release": ("current_only", "011", "host_profile", "111"),
    "normalized_matched": ("current", "host", "augur"),
}
COHORT_FIELDS = ("sid", "scene", "order", "cost", "y")
ROLES = ("validation", "test")
COUNTS = ("TP", "FP", "FN", "TN")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, context):
    check(np.allclose(actual, expected, atol=1e-10, rtol=0),
          f"{context}: {actual} != {expected}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_member(archive, entry, arms):
    member = entry["member"]
    blob = archive.read(member)
    check(sha256(blob).hexdigest() == entry["sha256"], f"Hash mismatch: {member}")
    with np.load(BytesIO(blob), allow_pickle=False) as data:
        names = (*COHORT_FIELDS, *arms)
        check(set(names) == set(data.files), f"Unexpected array keys: {member}")
        arrays = {name: data[name] for name in names}
    size = len(arrays["sid"])
    check(all(values.shape == (size,) for values in arrays.values()), f"Array shape: {member}")
    check(len(set(arrays["sid"])) == size and len(set(arrays["order"])) == size,
          f"Duplicate sessions or tie-break keys: {member}")
    check(np.isin(arrays["y"], [0, 1]).all() and (arrays["cost"] > 0).all(),
          f"Invalid labels or record counts: {member}")
    check(all(np.isfinite(arrays[name]).all() for name in arms), f"Nonfinite scores: {member}")
    return arrays


def ranking(y, score):
    """Noninterpolated AP and maximum test recall at a distinct-score 1% FPR boundary."""
    sorted_indices = np.argsort(-score, kind="stable")
    scores = score[sorted_indices]
    labels = y[sorted_indices]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    true_count = np.cumsum(labels)[ends]
    pred_count = ends + 1
    positives = int(true_count[-1])
    negatives = len(y) - positives
    check(positives > 0 and negatives > 0, "Ranking requires both classes")
    precision = true_count / pred_count
    ap = float(np.dot(np.diff(true_count, prepend=0) / positives, precision))
    eligible = (pred_count - true_count) <= negatives // 100
    recall = float(np.max(true_count[eligible], initial=0) / positives)
    return [ap, recall]


def threshold(y, score):
    negatives = np.sort(score[y == 0])[::-1]
    allowed = len(negatives) // 100
    value = float(np.nextafter(negatives[allowed], np.inf))
    check(np.count_nonzero(negatives >= value) <= allowed, "Validation cap exceeded")
    return value


def fixed_records(score, order, cost, allowance):
    selected = np.zeros(len(score), dtype=bool)
    remaining = allowance
    for index in np.lexsort((order, -score)):
        if cost[index] <= remaining:
            selected[index] = True
            remaining -= int(cost[index])
    return selected


def outcomes(y, selected):
    tp = int(np.count_nonzero(selected & (y == 1)))
    fp = int(np.count_nonzero(selected & (y == 0)))
    return dict(zip(COUNTS, (tp, fp, int(y.sum()) - tp, len(y) - int(y.sum()) - fp), strict=True))


def fold_budget_row(y, cost, selected, counts):
    return {
        "hits": counts["TP"],
        "negative_selected": counts["FP"],
        "picked_sessions": int(selected.sum()),
        "spent_records": int(cost[selected].sum()),
        "spent_positive_records": int(cost[selected & (y == 1)].sum()),
        "skipped_sessions": int((~selected).sum()),
    }


def record_seed(fold_counts, unique_ids):
    totals = Counter()
    for row in fold_counts:
        totals.update(row)
    totals["unique_positive"] = len(unique_ids)
    return dict(totals)


def check_original_budget_receipt(receipt, operating, matched):
    check(receipt["input_hashes"]["scores_complete"] == operating["source_complete_sha256"]
          == matched["original_sealed_source"]["completed_sha256"], "Matched source receipt mismatch")
    check(receipt["input_hashes"]["scores_validation_gate"] == operating["validation_lock_sha256"]
          == matched["original_sealed_source"]["validation_gate_sha256"], "Matched validation receipt mismatch")
    for rotation, repeated in ROTATIONS.items():
        reference = receipt["results"][rotation]
        cells = reference["fold_cells"]
        check(set(cells) == {cell for cell, key in CELLS.items() if key[0] == rotation},
              f"Matched record receipt cells: {rotation}")
        for arm in ("c_current", "c_host", "c_augur"):
            totals = []
            for seed in SEEDS:
                rows = [row for cell, row in cells.items() if CELLS[cell][2] == seed]
                regenerated = Counter()
                for row in rows:
                    check(row["eligible_records"] // 50 == row["budget_records"],
                          f"Matched record allowance: {rotation}/{seed}")
                    values = row["arms"][arm]
                    check(values["spent_records"] <= row["budget_records"],
                          f"Matched record spend: {rotation}/{seed}/{arm}")
                    regenerated.update(values)
                    regenerated.update({
                        "budget_records": row["budget_records"],
                        "eligible_records": row["eligible_records"],
                        "eligible_session_occurrences": row["n_test"],
                        "positive_occurrences": row["positive_occurrences"],
                    })
                saved = reference["seed_summaries"][str(seed)]["totals"][arm]
                for key, value in regenerated.items():
                    near(value, saved[key], (rotation, seed, arm, key))
                check(0 <= saved["unique_positive_sessions_selected"] <= saved["hits"],
                      f"Matched unique hits: {rotation}/{seed}/{arm}")
                totals.append(saved)
            summary = matched["results"][rotation]["arms"][arm]
            for key, value in reference["pooled_per_seed_means"][arm].items():
                near(np.mean([row[key] for row in totals]), value, (rotation, arm, key))
                near(value, summary["fixed_2pct_records"][key], (rotation, arm, key))
            op = operating["results"][rotation]["arms"][arm]
            near(np.asarray(op["mean"]) * 100,
                 [summary["AP_percent"], summary["test_rank_recall_at_1pct_fpr_percent"]],
                 (rotation, arm, "ranking receipt"))
            frozen = summary["validation_frozen_1pct"]
            near(np.asarray(op["validation_locked_micro_mean"][2:]) * repeated,
                 [frozen["mean_TP"], frozen["mean_FP"]], (rotation, arm, "frozen receipt"))


def check_precomputed_scene_rows(path, recomputed):
    rows = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (f'{row["protocol"]}_{row["fold"]}_{row["seed"]}', row["scene"])
            arm = row["arm"]
            check(arm not in rows[key], f"Duplicate matched scene receipt: {key}/{arm}")
            rows[key][arm] = {name: int(row[name]) for name in COUNTS}
    check(set(rows) == set(recomputed), "Matched scene receipt roster mismatch")
    for key, arms in rows.items():
        check(set(arms) == {"host", "augur"}, f"Matched scene arms: {key}")
        check(arms["augur"] == recomputed[key], f"Matched full-arm scene counts: {key}")
        for name in ("TP", "FP"):
            complementary = {"TP": "FN", "FP": "TN"}[name]
            check(arms["host"][name] + arms["host"][complementary]
                  == arms["augur"][name] + arms["augur"][complementary], f"Scene denominators: {key}")
    return rows


def verify(root):
    hashes = read_json(root / "SHA256.json")
    expected_files = {
        "scores.zip", "original_release_checked_summary.json", "normalized_matched_checked_summary.json",
        "original_matched_record_receipt.json", "original_matched_operating_export.json",
        "original_matched_scene_counts.csv", "original_matched_checked_summary.json",
    }
    check(set(hashes) == expected_files, "Unexpected files in SHA256.json")
    for file, expected in hashes.items():
        check(sha256((root / file).read_bytes()).hexdigest() == expected, f"Hash mismatch: {file}")

    release_reference = read_json(root / "original_release_checked_summary.json")["result"]
    normalized = read_json(root / "normalized_matched_checked_summary.json")
    normalized_reference = normalized["mode_comparison"]
    matched = read_json(root / "original_matched_checked_summary.json")
    matched_budgets = read_json(root / "original_matched_record_receipt.json")
    operating = read_json(root / "original_matched_operating_export.json")
    check_original_budget_receipt(matched_budgets, operating, matched)

    decisions = defaultdict(list)
    unique_hits = defaultdict(set)
    ranks = defaultdict(list)
    scene_frozen = {}
    test_appearances = defaultdict(Counter)
    cohort_baseline = {}
    role_membership = {}
    member_names = {"index.json"}
    cutoff_count = 0
    with ZipFile(root / "scores.zip") as archive:
        index = json.loads(archive.read("index.json"))["ensembles"]
        check(set(index) == set(ARMS), "Unexpected score ensembles")
        for tag, arms in ARMS.items():
            entries = index[tag]
            check(tuple(entries["arms"]) == arms and set(entries["cells"]) == set(CELLS),
                  f"Unexpected arms or cells: {tag}")
            if tag == "normalized_matched":
                check(entries["manifest_sha256"] == normalized["input_sha256"]["complete"],
                      "Normalized source manifest mismatch")
            for cell, (rotation, _, seed) in CELLS.items():
                entry = entries["cells"][cell]
                check(set(entry["segments"]) == set(ROLES), f"Unexpected roles: {tag}/{cell}")
                segments = {}
                for role in ROLES:
                    member = entry["segments"][role]
                    check(member["member"] == f"{tag}/{cell}.{role}.npz", f"Member path: {tag}/{cell}/{role}")
                    member_names.add(member["member"])
                    scores = load_member(archive, member, arms)
                    segments[role] = scores
                    members = set()
                    for sid, scene, order, cost, label in zip(
                            *(scores[name] for name in COHORT_FIELDS), strict=True):
                        key = int(sid)
                        members.add(key)
                        row = (str(scene), int(order), int(cost), int(label))
                        check(cohort_baseline.setdefault(key, row) == row, f"Cohort metadata mismatch: {key}")
                    check(role_membership.setdefault((cell, role), members) == members,
                          f"Cross-ensemble role parity: {tag}/{cell}/{role}")
                validation, test = (segments[role] for role in ROLES)
                val_scenes, test_scenes = set(validation["scene"]), sorted(set(test["scene"]))
                check(val_scenes.isdisjoint(test_scenes)
                      and len(val_scenes) == len(test_scenes) == ROTATIONS[rotation],
                      f"Invalid held-out scenes: {tag}/{cell}")
                check(role_membership[cell, "validation"].isdisjoint(role_membership[cell, "test"]),
                      f"Overlapping validation and test session IDs: {tag}/{cell}")
                test_appearances[tag, rotation, seed].update(role_membership[cell, "test"])
                y, cost = test["y"], test["cost"]
                allowance = int(cost.sum()) // 50
                check(set(entry["cutoff"]) == set(arms), f"Invalid cutoff keys: {tag}/{cell}")
                for arm in arms:
                    cut = threshold(validation["y"], validation[arm])
                    check(cut == entry["cutoff"][arm], f"Frozen cutoff mismatch: {tag}/{cell}/{arm}")
                    cutoff_count += 1
                    sent = test[arm] >= cut
                    selection = fixed_records(test[arm], test["order"], cost, allowance)
                    for mode, selection_mask in (("threshold", sent), ("fixed_records", selection)):
                        count_row = outcomes(y, selection_mask)
                        count_row["records"] = int(cost[selection_mask].sum())
                        key = (tag, rotation, arm, seed, mode)
                        decisions[key].append(count_row)
                        unique_hits[key].update(test["sid"][selection_mask & (y == 1)])
                    for scene in test_scenes:
                        mask = test["scene"] == scene
                        ranks[tag, rotation, arm, seed, str(scene)].append(ranking(y[mask], test[arm][mask]))
                        if tag == "original_release" and arm == "111":
                            scene_frozen[cell, str(scene)] = outcomes(y[mask], sent[mask])
                    if tag == "original_release" and arm == "111":
                        counts = fold_budget_row(y, cost, selection, outcomes(y, selection))
                        reference = matched_budgets["results"][rotation]["fold_cells"][cell]
                        check(counts == reference["arms"]["c_augur"], f"Matched full-arm record queue: {cell}")
        check(len(archive.namelist()) == len(member_names)
              and set(archive.namelist()) == member_names, "Unexpected zip members")
    check(cutoff_count == 560, f"Expected 560 verified cutoffs, found {cutoff_count}")
    check(len(cohort_baseline) == 11482 and len({r[1] for r in cohort_baseline.values()}) == 11482
          and sum(row[3] for row in cohort_baseline.values()) == 403
          and sum(row[2] for row in cohort_baseline.values()) == 2632790, "Cohort size or counts")
    scenes = sorted({row[0] for row in cohort_baseline.values()})
    for (tag, rotation, seed), counts in test_appearances.items():
        check(set(counts) == set(cohort_baseline) and set(counts.values()) == {ROTATIONS[rotation]},
              f"Test exposure mismatch: {tag}/{rotation}/{seed}")
    scene_rows = check_precomputed_scene_rows(root / "original_matched_scene_counts.csv", scene_frozen)
    for rotation in ROTATIONS:
        actual = Counter()
        for (cell, _), rows in scene_rows.items():
            if CELLS[cell][0] == rotation:
                actual.update(rows["host"])
        frozen = matched["results"][rotation]["arms"]["c_host"]["validation_frozen_1pct"]
        near([actual[c] / len(SEEDS) for c in COUNTS], [frozen[f"mean_{c}"] for c in COUNTS],
             (rotation, "host scene-count receipts"))

    result = {}
    for tag, arms in ARMS.items():
        result[tag] = {}
        for rotation, repeated in ROTATIONS.items():
            result[tag][rotation] = {}
            for arm in arms:
                row = {
                    "score_metrics": {},
                    "threshold": {},
                    "fixed_records": {},
                }
                result[tag][rotation][arm] = row
                seed_rank = []
                for seed in SEEDS:
                    ranked_by_scene = [ranks[tag, rotation, arm, seed, scene] for scene in scenes]
                    check(all(len(r) == repeated for r in ranked_by_scene), f"Ranking folds: {rotation}/{arm}/{seed}")
                    seed_rank.append(np.mean([np.mean(r, axis=0) for r in ranked_by_scene], axis=0))
                row["score_metrics"] = {
                    "scene_macro_AP": float(np.mean(seed_rank, axis=0)[0]),
                    "scene_macro_test_Recall_at_1pct_FPR": float(np.mean(seed_rank, axis=0)[1]),
                    "by_seed": {str(seed): val.tolist() for seed, val in zip(SEEDS, seed_rank, strict=True)},
                }
                for mode in ("threshold", "fixed_records"):
                    by_seed = {}
                    for seed in SEEDS:
                        key = (tag, rotation, arm, seed, mode)
                        by_seed[str(seed)] = record_seed(decisions[key], unique_hits[key])
                    row[mode] = {
                        "by_seed": by_seed,
                        "seed_mean": {k: float(np.mean([r[k] for r in by_seed.values()]))
                                      for k in next(iter(by_seed.values()))},
                    }
                    if tag == "normalized_matched":
                        expected = normalized_reference[mode][rotation][arm]["by_seed"]
                        check(by_seed == expected, f"Normalized operating counts: {rotation}/{arm}/{mode}")
                if tag == "original_release" and arm in release_reference[rotation]:
                    stats = row["threshold"]["seed_mean"]
                    expected = release_reference[rotation][arm]
                    near([stats[key] for key in COUNTS], [expected[f"mean_{key}"] for key in COUNTS],
                         (rotation, arm, "release operating counts"))
                if tag == "original_release" and arm == "111":
                    reference = operating["results"][rotation]["arms"]["c_augur"]
                    near(seed_rank, reference["seed_values"], (rotation, arm, "matched ranking by seed"))
                    expected = matched["results"][rotation]["arms"]["c_augur"]
                    for mode, values in (("threshold", expected["validation_frozen_1pct"]),
                                         ("fixed_records", expected["fixed_2pct_records"])):
                        actual = row[mode]["seed_mean"]
                        expected_keys = (dict(zip(COUNTS, (f"mean_{k}" for k in COUNTS), strict=True))
                                         if mode == "threshold" else
                                         {"TP": "hits", "FP": "negative_selected", "records": "spent_records",
                                          "unique_positive": "unique_positive_sessions_selected"})
                        for name, reference_name in expected_keys.items():
                            near(actual[name], values[reference_name], (rotation, arm, mode, name))

    return {
        "status": "PASS",
        "externally_hashed_artifact_files": len(hashes),
        "hashed_score_members": len(member_names) - 1,
        "recomputed_validation_cutoffs": cutoff_count,
        "raw_matched_recomputed_full_arm_scene_fold_seed_counts": len(scene_frozen),
        "raw_matched_recomputed_full_arm_record_queues": len(CELLS),
        "normalized_arm_rotation_seed_decision_totals": len(ROTATIONS) * len(ARMS["normalized_matched"]) * len(SEEDS) * 2,
        "scopes": {
            "original_release": "Score-level audit of released arms; 111 is the matched full arm.",
            "normalized_matched": "Score-level audit of all three normalized 900-tree main-head arms.",
            "original_matched_current_host": "Receipt-level checks only; raw current and host matched scores are not included.",
        },
        "results": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the extracted extended score bundle")
    parser.add_argument("--output", type=Path, help="Optional path for the recomputed JSON results")
    args = parser.parse_args()
    if args.output and args.output.resolve().parent == args.input_dir.resolve():
        check(args.output.name == "verification.json", "Use verification.json beside the input files")
    result = verify(args.input_dir)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PASS: 320 scored array members, 560 validation cutoffs, 80 matched full-arm queues")
    print("PASS: 120 matched full-arm scene rows and 60 normalized per-seed decision totals")
    print("Matched original-expression current/host controls: aggregate receipts only")


if __name__ == "__main__":
    main()

