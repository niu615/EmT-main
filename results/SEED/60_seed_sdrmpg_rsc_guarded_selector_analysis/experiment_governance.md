# Future Experiment Governance

Single-model exploration is frozen unless a proposed method is mechanistically different from the failed families and has explicit smoke-stop criteria.

Do not repeat:

- Backbone, attention, token, head, or capacity expansion.
- Explicit DG/alignment/source weighting/sampling methods.
- KD, consistency, R-Drop, or EMA-style regularization.
- Mixup, perturbation, pooled-feature repair, or feature-front-end replacement.
- Checkpoint soup, calibration, SWAD, SAM, or seed chasing.
- RSC variants that only retune the same weak/strong trade-off.

If a new idea is unavoidable, it must start from `feat/seed_sdrmpg_rsc`, use a new branch and result directory, pass engsmoke, weak smoke, and strong smoke before full, and be deleted immediately if it fails.
