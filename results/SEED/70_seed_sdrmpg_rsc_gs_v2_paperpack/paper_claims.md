# Paper Claims

- `07_seed_sdrmpg_rsc` is the single-model anchor with ACC `0.7963` and F1 `0.7852`.
- `27_seed_guarded_selector_paperpack` improves the anchor to ACC `0.8049` using source-validation guarded selection.
- `69_seed_sdrmpg_rsc_guarded_selector_v2` reaches ACC `0.8105` and F1 `0.8047`.
- Absolute ACC gain of `69` over `07`: `0.0142`.
- Absolute ACC gain of `69` over `27`: `0.0056`.
- The method remains source-only and uses no target labels, no TTA, and no target adaptation.
