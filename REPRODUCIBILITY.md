# Reproducing selected results from exported metrics

## Scope

The commands below check synthetic implementation contracts and recompute selected paper-table means from already exported per-scene or per-seed metrics. They do not train models, perform inference, or compute AP from sample-level predictions. They do not independently validate ground-truth labels, model checkpoints or upstream training. These distinctions apply to all numerical output.

## Environment and one-command check

Use Python 3.11 or newer; install numpy==2.3.5 from requirements.txt in an isolated environment. No GPU or input dataset is needed for these checks.

    python -m venv .venv
    # Linux/macOS: source .venv/bin/activate
    # Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python run_reproduction.py --output-dir ./reproduced_tables

The program runs state_124/state_contract_checks.py, test_current.py, and reproduce_core_tables.py sequentially. The first two tests use synthetic examples and check event-before-write, role-link, fusion, and related interface invariants; passing does not imply successful intrusion detection. The third reads frozen export files without modifying them, and checks 36 rounded AP/Recall@1% FPR paper rows against expected_core_tables.csv. Use a separate output folder; never write into results/.

## Data lineage and table coverage

| Output | Input included here | What is computed | Paper coverage |
|---|---|---|---|
| ait_tables_2_5_7.csv | results/ait_per_scene.csv | Average repeated holdouts per scene and seed, then average five seeds and eight scenes with equal weight; check against results/ait_aggregate.json. | All eight entries of Table 5; AUGUR and current-only rows of Table 2; available fallback rows of Table 7. Table 2 other baselines and Table 1 are not independently recomputed here. |
| autolabel_table_3.csv | results/autolabel_aggregate.json, rows | Check each 4-app/5-vulnerability group has each of three seeds; average equally and compare to the stored summary. | Table 3 means. Inputs are window-model metric exports, not window predictions. |
| cert_table_4.csv | results/cert_aggregate.json, score_file_audit | Mean five seed APs and five seed TP/1830 at the stated 1% budget. | Table 4; the files contain per-seed summaries, not individual activity scores. |

The expected values are transcribed from the manuscript, and are comparison targets rather than new ground truth. Each exported metric file is itself a research result requiring independent provenance verification. The 1% FPR value is the retrospective test-ranking budget metric defined in the paper; it is not a deployment threshold selected on the test set.

## AIT per-session score audit

The frozen AIT test-score files were checked separately with recompute_ait_scores.py: 80 NPZ files, 1,080 fold/seed/scene/arm rows, maximum absolute difference 3.33e-16 against the included per-scene CSV. This checks predictions against exported metrics, not model training or label correctness. A compact receipt is in results/ait_score_audit.json. The checked NPZ files contain the authors' predictions and are not included: obtaining AIT-ADS from its provider alone does not supply those predictions, and rebuilding them would additionally require the missing validated training pipeline and artifacts.

If the matching prediction files are available, independently run:

    python recompute_ait_scores.py --prediction-dir <directory-containing-npz> --reference-csv results/ait_per_scene.csv --output <new-directory>/ait_score_metrics.csv

Expected inputs are 80 files named {611|422}_fold_{01..08}_{42|1337|31415|2025|8675309}.test.npz. Each must contain one-dimensional arrays ids, y and score arrays 000 through 111 plus current_only, with aligned lengths; IDs have the scene prefix before the first underscore. Binary y and finite numeric scores are required. All ties in scores are evaluated together. Output is one CSV row per protocol/fold/seed/scene/arm, containing only AP and Recall@1% FPR, with no sample IDs, labels or individual scores. The script halts on missing files, invalid arrays, or a discrepancy exceeding 1e-10.

Both scene-holdout protocols use all eight scenes in the development history; different fold/seed runs are not eight independent blind deployments. The score audit does not expand that evaluation claim.

## Data, code and missing materials

AIT-ADS, AutoLabel and CERT r4.2 require obtaining the source data under the respective providers' conditions and licenses. The original data, all derived caches, trained checkpoints and complete adapted baseline pipelines are not included. Some snapshots/ scripts depend on historical experimental directories and should be treated as archived code, not end-to-end entry points. Exact raw-to-table reproducibility requires independently packaging and validating those dependencies, frozen split manifests, checkpoint provenance and sample-level predictions, subject to distribution permissions. This bundle makes no claim that those steps have been completed.

MANIFEST.sha256 covers the snapshot files and the updated repository README. The supplemental REPRODUCIBILITY.sha256 covers the new reproduction scripts, documentation, expected values, and inputs used in these checks. Both manifests use SHA-256 and paths relative to the repository root.
