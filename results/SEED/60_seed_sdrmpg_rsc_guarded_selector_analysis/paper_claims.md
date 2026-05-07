# SDRMPG-RSC-GS Analysis Pack

## Main Claim

SDRMPG-RSC-GS achieves 0.804924 ACC and 0.802264 F1 under the original SEED LOSO protocol.

This is a source-val guided guarded selector anchored on `07_seed_sdrmpg_rsc`; it is not a claim that a single `sdrmpg-rsc` checkpoint reaches 80.49%.

## Guarded Rule

1. Select between `01_seed_sdrmpg` and `07_seed_sdrmpg_rsc` by lower source-val NLL.
2. Allow `00_seed_paperfix` only when source-val NLL is lower than both `01` and `07`, source-val ACC is not lower than both, and source-val NLL is at most 0.32.
3. All selection signals are source-validation metrics; target-test labels are used only for final evaluation.

## Result Snapshot

| Method | ACC | F1 | Weak ACC | Strong ACC |
| --- | ---: | ---: | ---: | ---: |
| 07_seed_sdrmpg_rsc | 0.796296 | 0.785211 | 0.722727 | 0.913721 |
| 16_seed_sdrmpg_source_val_nll_selector | 0.800042 | 0.795256 | 0.749369 | 0.908880 |
| SDRMPG-RSC-GS / 27_seed_guarded_selector_paperpack | 0.804924 | 0.802264 | 0.762121 | 0.908880 |

## Safe Wording

Use: `SDRMPG-RSC-GS achieves 80.49% ACC under the original LOSO protocol.`

Avoid: `sdrmpg-rsc single model achieves 80.49% ACC.`
