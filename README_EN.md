# AUGUR research artifact

This repository provides source snapshots for event-preceding historical state,
entity features, controlled expression editing, local scoring, and score fusion,
along with selected exported metrics and fixed experimental configurations.

## Quick verification

Use Python 3.11 or newer and install the pinned dependency from requirements.txt.
Run:

```sh
python -m pip install -r requirements.txt
python run_reproduction.py --output-dir ./reproduced_tables
```

This runs two synthetic implementation-contract checks and recomputes selected
AIT, AutoLabel, and CERT table entries from exported aggregate metrics. It does
not train a model, read raw data, or independently validate labels or
checkpoints. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact
coverage, inputs, and limitations.

An additional [AIT score-audit summary](results/ait_score_audit.json) reports
checks performed on the authors' per-session scores; those score files are not
included in this repository.

## Contents

- `state_124/`: cumulative state implementation and synthetic checks.
- `current/`: direct-entity readout and score-fusion implementation.
- `snapshots/`: selected training, expression-editing, and local-scoring code.
- `configs/` and `results/`: experiment configurations and exported summaries.
- `MANIFEST.sha256` and `REPRODUCIBILITY.sha256`: file-integrity checks.

## Data and reproducibility boundary

AIT-ADS, AutoLabel, and CERT r4.2 data must be obtained from their respective
providers and used in accordance with their licenses. Raw data, full derived
caches, trained model checkpoints, per-session predictions, and complete
adapted baseline pipelines are not included. Some historical scripts require
those missing inputs, so this snapshot does not yet support independent
raw-data-to-table reproduction.

The eight AIT scenes participated in method development. AutoLabel and CERT
use task-specific state adaptations, not the complete AIT three-branch model.
Their evaluation units differ. The retrospective Recall@1% FPR statistic is a
test-ranking budget, not a frozen deployment threshold.
