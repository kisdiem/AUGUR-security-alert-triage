# Inventory of the original AIT training chain

This is an audit of the original research workspace, not an end-to-end release. The current repository does not contain every input listed below.

## Top-level execution and dependencies

The latest 80-fold/seed scoring run is orchestrated by the historical snapshot ait_agreement_augur_paper_20261115_v1/run.py (also provided in snapshots/). It reads sealed predictions and models from prior stages, uses dynamically imported earlier run modules, and fits only some heads; it does not by itself retrain the full pipeline from source alerts.

An observed dependency chain is:

1. The agreement run loads ait_entity_roles_paper_20261115_v1/run.py.
2. The role run loads ait_422_entity_redesign_20261115_v1/run.py and sealed output from ait_augur_action_contrast_20261115_v1.
3. The redesign loads ait_422_expression_shift_20261115_v1/fit.py and a donor feature bundle.
4. The expression stage loads prior contrast models and uses prepare.py and external normalization/cache code.
5. The contrast stage loads ait_full_same_fusion_20261115_v1/run_full.py; this imports ait_augur_host_entity_20261114_v1/capacity.py.
6. The capacity stage dynamically loads ait_5seed_confirmation_20261113_v1/run_five.py and sealed host baselines; run_five loads 6/1/1 context stages and ait_augur_rotation422_20261113_v3/run_rotation422.py.
7. The rotation script reads AIT source caches, a legacy cross-scene session manifest, precomputed protocol evidence and embedding/feature banks.

The historical scripts use absolute machine paths, dynamic imports and integrity locks containing exact SHA-256 values. Moving or editing a pinned script changes its checksum. Making them portable requires a new, explicitly versioned configuration and lock-generation process; simply substituting path strings and bypassing checks would not establish equivalence.

## Observed workspace sizes

At audit time the historical 5-seed stage was approximately 620 MiB, host baseline 219 MiB, host/entity stage 658 MiB, two history stages 400 and 263 MiB, action-contrast stage 483 MiB, role stage 796 MiB, and final agreement output 139 MiB. These totals include models, intermediate scores and other nonessential files; they are not a measured minimal redistributable package. The AIT scene-cache subtree was approximately 2.1 GiB. The parent cache workspace was approximately 12 GiB and includes unrelated or redundant artifacts; two older source roots were approximately 843 MiB and 1.9 GiB. No conclusion about redistribution rights follows from size or availability.

The final agreement output contains 80 trained model files and 160 validation/test NPZ score files. Some other stages contain additional model heads. A full inference or training bundle must identify all required heads and feature definitions per branch; final-stage files alone are insufficient. Model loading based on joblib/pickle must only be performed from trusted sources.

## What is still required

- Determine and legally distribute or regenerate all required derived caches, source inputs and MiniLM model files.
- Export frozen split manifests, all pre-processing and classifier configurations, and checksums of the actual inputs used.
- Audit and sanitize source, configurations and serialized models for personal paths, credentials and identifying metadata before sharing.
- Recreate an isolated runtime from an archived environment or validate a new lock file. Versions of the currently running server environment are not proof of the versions used to train historical models.
- Refactor paths without silently disabling historical integrity checks; provide separate original and portable manifests.
- Start with a clean environment and independently execute raw input preparation, state generation, training, validation-only model selection and final test scoring; compare generated predictions to the frozen reference.

Until those steps have actually been performed, the score-level and aggregated-metric checks documented in REPRODUCIBILITY.md are the verified scope.
