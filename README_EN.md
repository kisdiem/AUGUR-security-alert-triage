# AUGUR: maintained historical evidence for security-alert session ranking

The historical snapshots remain unchanged. Portable, versioned entry points are
provided under `portable/`, with original prepared numerical fixtures and saved
validation/test scores. Start in a fresh directory and install the pinned runtime:

```sh
python -m pip install -r requirements-reproduction.txt
python run_reproduction.py --output-dir reproduced --include-training
```

The run checks state contracts and streaming invariants; 36 archived metric rows;
240 original validation cutoffs; 80 entity-switch cells; 80 original-expression
matched-control cells; and frozen numerical inference for 3,520 sessions. The
training option fits all three main forests and three protocol experts for the
supplied 4/2/2 fold/seed from original prepared matrices, seals validation cutoffs,
then compares test predictions with the frozen reference. It does not select a
configuration using test results. Omit the flag for score/inference checks only.

See [portable/README.md](portable/README.md) for separate commands and provenance.
This is not a claim of all-fold retraining or raw-data-to-table reproduction.
Raw text preparation, encoding and independent label reconstruction, other
baseline pipelines, external sample-score releases and full service replay remain
outside the portable release. Historical input snapshots are supplied for
traceability, not represented as standalone raw-data entry points.
