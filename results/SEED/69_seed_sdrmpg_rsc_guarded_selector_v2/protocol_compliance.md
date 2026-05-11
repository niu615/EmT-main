# Protocol Compliance

- `valid_source_only`: `true`
- No LOSO change.
- No target label usage.
- No TTA or target adaptation.
- No validation split or checkpoint-rule modification.
- Candidate pool is fixed to `00_seed_paperfix`, `01_seed_sdrmpg`, and `07_seed_sdrmpg_rsc`.
- `07_seed_sdrmpg_rsc` remains the default anchor; switching requires source-validation evidence only.
