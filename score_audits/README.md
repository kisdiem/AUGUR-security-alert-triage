# Additional sample-score audits

This directory contains aggregate-only audit outputs and portable verification
scripts. The source prediction files are not included and must be supplied
separately. These audits recompute metrics from model-produced scores; they do
not independently verify labels, train models, or reprocess raw datasets.

CERT: four conditions and five seeds (20 test score files) were checked.
The script excludes five unverified activities, computes average precision and
recall at FPR <= 1%, and compares AP to stored per-seed metrics. The score NPZ
files contain user/case encodings and timestamps and are not included.

AutoLabel: current-input and state-adapted versions across five held-out CVE
groups and four held-out application groups, using three seeds, were checked
(54 configurations). The script recomputes pooled-window AP and recall and
compares AP to stored per-seed metrics. Other baselines were not audited at
the sample-score level. Window scores and labels are not included.

Install requirements-score-audit.txt in a separate environment and consult
the scripts' --help for expected directory structure and output. The source
prediction files and baseline checkpoints are not publicly distributed in
this artifact. Review applicable data and derivative-data licensing before
redistributing any inputs.
