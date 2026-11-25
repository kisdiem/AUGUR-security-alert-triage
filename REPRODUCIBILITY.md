# Reproduction layers and verified scope

Use `requirements-reproduction.txt` for the portable release. NumPy-only historical
score scripts remain usable separately. The full current entry is:

```sh
python run_reproduction.py --output-dir reproduced --include-training
```

| Layer | Entry/input | What it checks | What it does not establish |
|---|---|---|---|
| State | `state_124/state_contract_checks.py`, `test_current.py`, `portable/test_replay.py` | Read-before-write, persistence, role facts, label independence and implementation contracts on explicit synthetic streams | Detection accuracy or real raw-data replay |
| Original scores | `ait_saved_score_audit/` | 80 cells, 240 validation cutoffs and original metric/queue receipts | Training, inference or independent ground truth |
| Entity switch | `portable/data/entity_switch.zip` | 160 validation/test arrays, 160 cutoffs, exact registered fusion, AP and Recall@1% FPR | Removing all entity inputs or refitting each arm |
| Matched controls | `portable/data/matched_controls.zip` | 160 original-expression arrays, 240 cutoffs, full/current/archived-host scores and deterministic record queues | New performance experiments or original ordered-list receipt equivalence |
| Frozen inference | `portable/fixtures/numeric422/` | Actual 900-main-tree +160-expert-tree scoring for 3,520 test sessions | Raw embedding, label reconstruction or all-fold inference |
| Numerical training | same fixture, `portable/train.py` | One 4/2/2 fold/seed, all six estimators, original prepared edits and recorded parameters; validation seal before test scoring | All 80-cell training, raw preprocessing or 6/1/1 context-head retraining |
| Archived tables | `reproduce_core_tables.py` | 36 rows from exported AIT, AutoLabel and CERT metric summaries | External task inference or full upstream-baseline reproduction |

Every score archive has a SHA-256 manifest. Every prepared numerical training
fixture has an input manifest. Runtime code rejects a changed score archive,
train/validation session overlap, wrong shapes and unrecognized frozen software
versions. The dependency lock is the tested portable environment, not an assertion
that every older experiment used that environment.

## Statistics and thresholds

AP is the sum of precision weighted by recall increments, with tied scores admitted
together. Recall@1% FPR is the best test-ranking recall under a 1% negative budget.
It is separate from frozen operational outcomes. Operational cutoffs are computed
from validation negatives using the original strict boundary and are then applied
to test scores. Repeated 4/2/2 appearances are averaged within scene/seed before
the equal-scene mean; standard deviations concern five seed means.

The record queue allows floor(2% of test record volume) per fold. Sessions sort by
descending score then ID; a session that does not fit is skipped. The verifier
saves ordered selected IDs and their digest, rather than claiming that equal
counts prove equal historical membership. Unique-positive counts take the union
within each seed. Labels follow the benchmark policy, not independent adjudication
of all policy-negative activity.

## Independent raw reconstruction and other methods

Obtain provider data under its license. An independent raw-to-table run would
also need the raw source-to-session and label mapping, embedding banks and training
weights, full controlled-view preparation and every upstream adaptation.
`TRAINING_DEPENDENCIES.md` inventories that original chain. The numerical fixture
and portable entries close selected gaps; they do not make that chain disappear.
Historical table-numbered CSV names are retained for compatibility; use result
group names and the current paper's row labels when comparing revisions.

AutoLabel/CERT exported metric summaries and optional scripts remain in
`score_audits/`; their undistributed sample-level inputs are not newly released.
The optional older `--score-bundle` flag still verifies the separately supplied
extended archive, including normalized controls. The two new original-expression
archives are included directly in this repository.

## Integrity

`MANIFEST.sha256` and `REPRODUCIBILITY.sha256` preserve and refresh the corresponding
documented file sets. `portable/SHA256SUMS` covers all files in the new portable
subtree other than itself. None is evidence of independently correct labels.
