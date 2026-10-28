# Notebook guide

Filenames below match the supplied files. Branches have different input conventions and are not a single sequential benchmark.

| Notebook | Purpose |
|---|---|
| `01_exploratory_analysis.ipynb` | Initial dataset exploration |
| `02_preprocessing_signal_cleaned.ipynb` | Historical 8–30 Hz preprocessing and full-recording ICA |
| `03_mi_decoding_CSP_LDA_.ipynb` | Binary, within-subject CSP–LDA with repeated trial splits |
| `04_bci_simulation.ipynb` | A07 CSP–LDA illustration; not held-out validation |
| `05.0.preprocessing_signal_dl.ipynb` | Historical 1–40 Hz preprocessing for neural experiments |
| `06.bci_simulation.ipynb` | EEGNet inference with simulated LSL replay |
| `08.0_deep_learning_guassian_class_sub_holdout.ipynb` | Fixed 5/2/2 participant holdout and validation ablations |
| `08.1_nested_loso.ipynb` | Nine-fold EEGNet LOSO with three-fold participant-grouped inner selection |
| `09.0_attention_based_subwise.ipynb` | Attention-model fixed holdout and seed comparisons |
| `09.1_attention_based_loso.ipynb` | Attention-model LOSO with designated validation participants |
| `10.0_Riemannian_baselines_LOSO.ipynb` | Covariance classifiers and historical neural-table comparisons |
| `10.1_Riemannian_transductive_adaptation_LOSO.ipynb` | Unlabeled target alignment and tangent-reference adaptation |
| `11.0_preprocessing_both_sessions.ipynb` | Revised GDF loading, calibration-only ICA, cue/run/channel audit |
| `11.1_cross_session_eegnet.ipynb` | Train/validate on T, test on E of the same participant |
| `11.2_interpretability_mu_beta.ipynb` | Frozen cross-session model perturbations, controls and ERD/ERS |

## Execution branches

- **Historical binary:** 01 → 02 → 03; 04 is an illustration.
- **Historical cross-participant neural:** 05 → 08.0 or 08.1; the 09 attention experiments use their own splits.
- **Riemannian:** 10.0 → 10.1, with the expected preprocessed inputs and optional historical neural CSVs.
- **Revised cross-session:** 11.0 → 11.1 → 11.2. The final notebook requires saved configuration, cleaned files, predictions and model checkpoints from this branch.
- **Replay demos:** 04/06 and application scripts use their own inputs and are not validation steps.

## Paths and environments

Check the paths at the top of each notebook before running it. In particular, the supplied notebook 05 and 08/09 loaders do not all use the same order of `clean`, `raw` and `1_40` in filenames. Documentation cannot resolve a missing file by silently treating different preprocessing artifacts as interchangeable. Internal comments referring to older notebook numbers also remain in the cross-session sources; use the actual filenames in the table above.

The code spans TensorFlow/Keras and PyTorch/Braindecode as well as MNE and pyRiemann. Record the versions used by each branch and test dependency files before claiming one environment reproduces every historical experiment. No tested universal environment is supplied by this documentation update.
