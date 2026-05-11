# Analysis Notes

## What v2 Fixes
The v2 rule primarily avoids overly aggressive switches away from the `07` anchor. It improves over result `27` on `5` subjects and is worse on `1` subject.

Key repaired folds include `sub4`, `sub10`, and `sub11`, where result `27` switched to `01_seed_sdrmpg` but v2 safely falls back to `07_seed_sdrmpg_rsc`.

## Remaining Limitation
`sub7` loses a small amount versus result `27` because v2 is more conservative, but the aggregate ACC and strong-subject protection improve.

## Sensitivity
The local sensitivity grid contains `27` neighboring threshold settings, and `27` pass the main gates. This supports that v2 is not a single-threshold accident.
