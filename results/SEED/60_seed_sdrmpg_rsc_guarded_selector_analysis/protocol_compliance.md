# Protocol Compliance Statement

- LOSO protocol is unchanged.
- Target labels are never used for model selection.
- No target adaptation, no TTA, and no target-test feedback are used.
- Source-validation metrics are the only selector inputs.
- The protected `07_seed_sdrmpg_rsc`, `17_seed_guarded_3way_source_val_selector`, and `27_seed_guarded_selector_paperpack` result directories must not be overwritten.
- Test labels are used only once for final evaluation and reporting.
