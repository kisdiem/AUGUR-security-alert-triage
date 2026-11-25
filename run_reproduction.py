"""Check state contracts, published score decisions, and archived table metrics."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for verification receipts and CSV summaries")
    parser.add_argument("--score-bundle", type=Path,
                        help="Directory containing the extracted extended AIT score archive")
    parser.add_argument("--include-training", action="store_true",
                        help="Also retrain all six estimators in the supplied 422 numerical cell")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    protected = [ROOT / p for p in ("results", "portable/data", "portable/fixtures", "snapshots")]
    if out == ROOT or any(out.is_relative_to(p.resolve()) for p in protected):
        parser.error("choose a new output directory outside archived results, fixtures and sources")
    if args.score_bundle:
        bundle = args.score_bundle.resolve()
        if not (bundle / "SHA256.json").is_file():
            parser.error("--score-bundle must point to the extracted archive containing SHA256.json")
        if out == bundle:
            parser.error("choose an output directory separate from the score bundle")
    out.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("state_124/state_contract_checks.py", []),
        ("test_current.py", []),
        ("portable/test_replay.py", []),
        ("ait_saved_score_audit/verify_minimal_ait_repro.py",
         ["--output", str(out / "ait_saved_score_receipt.json")]),
        ("reproduce_core_tables.py", ["--output-dir", str(out)]),
        ("portable/audit_scores.py", ["--kind", "switch", "--archive", str(ROOT / "portable/data/entity_switch.zip"),
                                     "--output-dir", str(out / "entity_switch")]),
        ("portable/audit_scores.py", ["--kind", "matched", "--archive", str(ROOT / "portable/data/matched_controls.zip"),
                                     "--output-dir", str(out / "matched_controls")]),
        ("portable/infer.py", ["--bundle", str(ROOT / "portable/fixtures/numeric422"),
                              "--output-dir", str(out / "frozen_inference")]),
    ]
    if args.include_training:
        tasks.append(("portable/train.py", ["--bundle", str(ROOT / "portable/fixtures/numeric422"),
                      "--output-dir", str(out / "numeric_retraining"), "--verify-test"]))
    if args.score_bundle:
        tasks.append(("score_audits/verify_extended_ait.py",
                      ["--input-dir", str(bundle), "--output", str(out / "ait_extended_score_receipt.json")]))

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONOPTIMIZE", None)
    for script, arguments in tasks:
        print(f"\nRUN: {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, env=env, check=True)
    print("\nPASS: state contracts, original matched controls, entity-switch replay and frozen numerical inference.")
    print("Numerical one-cell retraining: " + ("included" if args.include_training else "not requested"))
    print("Scope: published scores and prepared numerical features. Raw preprocessing, all-fold retraining and deployment remain outside this run.")


if __name__ == "__main__":
    main()
