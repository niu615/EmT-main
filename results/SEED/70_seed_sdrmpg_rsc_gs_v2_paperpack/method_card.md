# SDRMPG-RSC-GS-v2 Method Card

## Core Idea
SDRMPG-RSC-GS-v2 treats `sdrmpg-rsc` as the default reliability anchor and switches to an alternative candidate only when source-validation evidence is strong enough.

## Selection Rule
1. Default model: `07_seed_sdrmpg_rsc`.
2. Select `00_seed_paperfix` only if source-val NLL is lower than both `01` and `07`, source-val ACC is not lower than both, and source-val NLL is at most `0.32`.
3. If `00` is not allowed, select `01_seed_sdrmpg` only if it satisfies the ACC-margin/NLL-guard condition or the hard-source-val-fold condition.
4. Otherwise fall back to `07_seed_sdrmpg_rsc`.

## Main Result
`SDRMPG-RSC-GS-v2` achieves `81.05%` ACC and `80.47%` F1 under the original SEED LOSO protocol.

## Correct Claim
Use: `SDRMPG-RSC-GS-v2 achieves 81.05% ACC under source-validation guided model reliability selection.`

Do not use: `A single sdrmpg-rsc checkpoint achieves 81.05% ACC.`
