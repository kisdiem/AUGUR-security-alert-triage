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
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == (ROOT / "results").resolve():
        parser.error("choose an output directory outside results/")
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
        ("ait_saved_score_audit/verify_minimal_ait_repro.py",
         ["--output", str(out / "ait_saved_score_receipt.json")]),
        ("reproduce_core_tables.py", ["--output-dir", str(out)]),
    ]
    if args.score_bundle:
        tasks.append(("score_audits/verify_extended_ait.py",
                      ["--input-dir", str(bundle), "--output", str(out / "ait_extended_score_receipt.json")]))

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONOPTIMIZE", None)
    for script, arguments in tasks:
        print(f"\nRUN: {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, env=env, check=True)
    print("\nPASS: implementation contracts, frozen AIT score decisions, and exported-metric table rows.")
    print("Scope: synthetic contracts, saved predictions and archived metrics. Raw preprocessing, training and deployment are outside this run.")


if __name__ == "__main__":
    main()
