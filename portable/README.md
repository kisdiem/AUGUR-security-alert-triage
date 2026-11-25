# Portable historical state and numerical reproduction

These entries are a versioned port of the selected interface and classifier path.
They do not replace or silently edit the historical snapshots.

## 1. State replay from adapted observations

```sh
python portable/replay.py --input adapted_observations.jsonl --output replayed_state.npz
```

Each input line has `session_id` (or null), `event`, and optional `close_session`.
`event` uses the adapter schema: integer `timestamp_ns`, `event_id`, `action`,
visible actor/target/account/host keys, `focus`, `entity_observations`, and observed
outcome/parent fields when available. Rows must already follow the sealed source
order; equal timestamps preserve input order. This is not a parser for arbitrary
provider files. The existing AIT adapter remains in `state_124/implementation/`.

No label field is read. All observations update state, including those without a
supervised session. A session reads state before the current write; closure emits
its pooled features but does not reset historical state. Outputs are 124 host,
124 entity, 54 role and 24 relation dimensions. Current-text embeddings must be
prepared separately. The six explicit replay tests are synthetic contract tests.

## 2. Direct-entity switch and matched controls

```sh
python portable/audit_scores.py --kind switch --archive portable/data/entity_switch.zip --output-dir switch_check
python portable/audit_scores.py --kind matched --archive portable/data/matched_controls.zip --output-dir matched_check
```

Each archive has all 80 fold/seed cells and both partitions. Switch fields include
the cumulative anchor, host/direct estimates and visible-support flags; the code
reconstructs complete AUGUR exactly. The off score still contains cumulative
entity history. Local/context increments are already present in the frozen branch
scores. Matched-control scores preserve the original 900-tree experiment; they
are not refitted or substituted for the switch comparison.

Receipts report AP, retrospective Recall@1% FPR, validation-frozen decisions,
record-budget counts, unique positives and exact newly reconstructed ordered queues.
They are score audits, not independent training or label validation.

## 3. Frozen numerical inference

```sh
python portable/infer.py --bundle portable/fixtures/numeric422 --output-dir inference_check
```

The fixture supplies standard scikit-learn estimators, original numerical inputs
and expected test scores for 422/fold 01/seed 42. Input dimensions and all three
branch views are checked. The 54/53/53-tree experts adjust the three 300-tree heads;
the registered correction produces the final score. Numerical identity is reported
separately from the 1e-12 acceptance tolerance. No text or labels are reconstructed.

Joblib/pickle is executable content: use only the trusted, hash-pinned supplied
checkpoint. A matching hash establishes identity, not safety of an arbitrary file.
The source estimators require scikit-learn 1.8.0. The tested portable dependency
versions are pinned in `requirements-reproduction.txt`.

## 4. Numerical training with a validation seal

```sh
python portable/train.py --bundle portable/fixtures/numeric422 --output-dir training_check --verify-test
```

`train.npz` has the original 5,507 prepared training views, including 4,885 natural
sessions; history prefixes shared by branch matrices are packed once without
changing coordinates, values or row order. `validation.npz` contains 3,077 sessions.
Expert inputs and category-union targets are drawn only from natural training rows.
All three main forests and all three experts are fitted anew with the parameters
of the recorded cell. No context head is used in 4/2/2. The models and validation
cutoffs are written before optional test inference. Original checkpoints are not
overwritten; no test labels select training or thresholds.

This is a prepared-feature reproduction of one cell. It does not recreate the
raw-to-embedding pipeline, official label alignment, every controlled edit from
raw messages, all 80 training cells, or the 6/1/1 context-head fits. The frozen input
manifests distinguish source artifacts from this portable packing.

## Attribution and visibility

The numerical features and scores derive from the AIT Alert Data Set, Max Landauer,
Florian Skopik and Markus Wurzenberger, Zenodo 2023, DOI 10.5281/zenodo.8263181,
CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
Changes include session grouping, study-specific policy labels, numerical feature
generation, estimator training and score export. This is not the original dataset.
The archives include the original attribution notice. No raw messages or host keys
are included. Embeddings and derived identifiers can remain linkable and are not
claimed comprehensively anonymous. Third-party software retains its own licenses.
The original source snapshots and dependency inventory remain separately scoped.
