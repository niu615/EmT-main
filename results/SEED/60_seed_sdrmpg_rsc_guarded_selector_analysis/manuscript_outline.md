# Manuscript Outline: SDRMPG-RSC-GS

## Core Thesis

The main limitation of `sdrmpg-rsc` on SEED is not insufficient representation capacity. The model already achieves high strong-subject accuracy, but weak subjects exhibit a large source-val to target-test gap. SDRMPG-RSC-GS addresses this by modeling subject-dependent candidate reliability using source-validation signals only.

## Suggested Contributions

1. A strong `sdrmpg-rsc` anchor candidate built on TGC, RMPG, TCT, and RSC.
2. A source-val guided guarded selector that defaults to `sdrmpg-rsc` and only switches when validation evidence is strong.
3. A weak/strong subject boundary analysis showing why average-risk single-model tuning repeatedly fails.
4. A negative-result taxonomy that explains why common DG, normalization, loss, checkpoint, and seed-chasing variants are unreliable under this protocol.

## Main Result

`SDRMPG-RSC-GS` achieves 0.804924 ACC and 0.802264 F1 under the original SEED LOSO protocol, with weak mean 0.762121.

## Required Wording

Use: `SDRMPG-RSC-GS achieves 80.49% ACC under the original LOSO protocol.`

Avoid: `sdrmpg-rsc single model achieves 80.49% ACC.`
