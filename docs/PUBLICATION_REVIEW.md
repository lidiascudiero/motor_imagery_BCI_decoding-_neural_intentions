# Publication review of the supplied BCI files

This note records the limits found while preparing a clearer README. It is a source/output review, not a rerun or certification of the complete pipeline. The original downloads have not been modified.

## Corrected reporting

1. **77% binary result:** traced to A01, not an aggregate over all nine participants. The nine-participant mean is 65.7% accuracy, SD 8.4 percentage points. A07's mean is 77.9%.
2. **Cross-session SD:** NoAug BA is 56.1% ± **13.6%**; Aug is 56.1% ± **13.4%**, as displayed in notebook 11.1.
3. **New experiments have results:** cross-session and mu/beta notebooks are executed; they should not be described merely as future work.
4. **Existing good wording retained:** the supplied README already qualifies fairness, topographies and replay in several places. Further revision should sharpen those boundaries rather than imply that every older critique still applies verbatim.

## Implementation and interpretation items still to address

- **Binary app:** the supplied script uses true labels for arrows and random values for confidence. Either keep the accurate UI-simulation description or implement genuine saved-model inference. This documentation update does not implement that change.
- **Historical ICA:** preprocessing fits a participant's full recording before later splits. Describe the target-signal access; do not assert that all preprocessing was learned only on model-training data.
- **Historical channel definitions:** the manually assigned montage differs from the corrected official list in notebook 11.0. CSP output logs also show 25 channels. Audit channel selection and naming before strong EEG-only or anatomical claims; rerun affected results if the implementation is corrected.
- **Validation terminology:** notebook 08.1 contains a claim of an “external/clinical validation step.” Nested resampling on this dataset does not establish external or clinical validity. It also mentions exactly compute-matched training, while outer final epoch counts can differ between conditions.
- **Historical result imports:** notebooks 10.0/10.1 import earlier neural CSVs; their EEGNet numbers are not the latest nested-LOSO results. Preserve that distinction or rebuild a deliberately matched comparison.
- **Preprocessing filenames:** notebook 05's saved filenames and later loaders differ. Resolve the actual artifact provenance rather than merely renaming an unknown file.
- **Demo checkpoint provenance:** establish which training run produced the deployed checkpoint. Do not assign the 63.4% historical score or claim that A07 was unseen from a shared filename alone.
- **Four-class app duplication:** the supplied `7.bci_demo.py` contains duplicated application code. Consolidate it before naming it as a canonical entry point.
- **Cross-session calibration:** the default fits test-session ICA on that session's unlabeled EOG-calibration block. This is legitimate to describe as calibration-assisted evaluation, not zero-calibration transfer.
- **Interpretability:** mu dependence is supported for four participants under the specified controls. Beta dependence and selective motor-cortex localization are not general findings. The variance-preserving variant is post hoc; trial-bootstrap intervals are not participant-level confidence intervals.

## Before publishing

Keep primary result tables and their notebooks traceable. Treat the corrected 11.0–11.2 branch separately from older preprocessing. Use actual saved plots, avoid absolute personal paths in screenshots and logs, and keep raw EEG and derived participant data out of the documentation bundle unless distribution terms have been checked. This bundle includes only documentation and one existing notebook figure.
