# Methods and results by evaluation setting

This document separates reported experiments by task, evaluation unit and preprocessing. 

## 1. Binary CSP–LDA: resolving the 77% versus 65.7% figures

Notebook `03_mi_decoding_CSP_LDA_.ipynb` evaluates left versus right imagery with ten stratified ShuffleSplit iterations per participant, using an 80/20 training/test split. CSP and LDA are in a scikit-learn pipeline, so CSP is fitted separately inside each training split.

| Reported value | Actual scope |
|---|---|
| Mean accuracy 0.657; between-subject SD 0.084 | Mean of the nine participant-specific CV estimates |
| Mean accuracy 0.769 | A01T only, across ten repeated splits |
| Aggregated accuracy 0.769; BA 0.769 | A01T only: 290 test-prediction records from repeated splits |
| Mean accuracy 0.779 | A07T only, across ten repeated splits |


## 2. Fixed-participant holdout EEGNet

`08.0_deep_learning_guassian_class_sub_holdout.ipynb` uses five training, two validation and two test participants on the four-class task. Test participants are A02T and A09T: 576 trials in total, 144 per class.

- Overall test accuracy: **0.470**.
- A02T accuracy: **0.382**.
- A09T accuracy: **0.559**.

The small test cohort makes this result sensitive to participant assignment. The three-seed band/augmentation ablation in that notebook reports validation-subject results, not additional independent test replications. Its 1–40 Hz NoAug validation accuracy is 0.531 ± 0.025; augmentation is 0.534 ± 0.024; 8–30 Hz is 0.350 ± 0.014.


## 3. Nested LOSO EEGNet

`08.1_nested_loso.ipynb` implements nine outer test folds with three-fold GroupKFold inside the eight development participants. It compares two learning rates, using shared hyperparameter selection across NoAug and Aug. Final model training uses all eight development participants, with epoch counts derived from inner validation.

Input is 22 selected EEG channels, 751 samples per epoch at 250 Hz, from previously cleaned session-T files. Whole-window, per-channel z-scoring is used; it is an offline epoch transformation and differs from the pre-cue normalization of the cross-session experiment. The historical preprocessing boundary above still applies.

| Metric | NoAug | Aug |
|---|---:|---:|
| Accuracy | 0.553 ± 0.090 | 0.553 ± 0.085 |
| Balanced accuracy | 0.553 ± 0.090 | 0.553 ± 0.085 |
| Macro F1 | 0.540 ± 0.097 | 0.542 ± 0.088 |
| Cohen's kappa | 0.404 ± 0.120 | 0.404 ± 0.113 |

All nine subjects contribute 288 test trials: 2,592 trials per condition. The standard deviations above describe variation across participants. NoAug BA spans approximately 0.417–0.681.

Gaussian noise has sigma 0.05 in z-scored units, applied to half the training trials on average. It is not applied to validation or test data. The experiment matches architecture, seeds, shared hyperparameters and dataset size. However, final epoch counts can differ by condition, so final optimization-update counts are not necessarily identica.

Nested CV on an already explored dataset is internal resampling validation, not independent external or clinical validation. Earlier dataset exposure and multiple experiments also limit claims of fully blinded discovery.

## 4. Attention-based experiments

`09.0_attention_based_subwise.ipynb` reports fixed-holdout BA of **0.491319** without augmentation and **0.465278** with augmentation. A separate three-seed evaluation reports means **0.483** and **0.499**; do not combine these distinct summaries or infer a consistent augmentation effect.

`09.1_attention_based_loso.ipynb` uses an outer test participant and a designated validation participant, rather than the full three-fold inner search in notebook 08.1. Its output reports BA **0.524 ± 0.114** without augmentation and **0.520 ± 0.122** with augmentation. It is not a matched architecture comparison against the nested EEGNet result.

## 5. Riemannian baselines and adaptation

`10.0_Riemannian_baselines_LOSO.ipynb` reports inductive MDM and tangent-space logistic regression BA means from **0.263 to 0.281**. These results characterize this implementation, feature representation and preprocessing; they do not establish that Riemannian methods are generally unsuitable for motor imagery.

`10.1_Riemannian_transductive_adaptation_LOSO.ipynb` reports, among other variants:

| System | Target access | BA mean ± SD |
|---|---|---:|
| MDM, no normalization, static | No target adaptation | 0.263 ± 0.029 |
| MDM, no normalization, RA-calib48 | Unlabeled 48-trial target calibration | 0.330 ± 0.049 |
| MDM, no normalization, RA-full | Full unlabeled target trial set | 0.336 ± 0.052 |
| Tangent-space logistic regression, no normalization, tsupdate | Target-dependent reference update | 0.333 ± 0.060 |

## 6. Within-subject, cross-session EEGNet

`11.0_preprocessing_both_sessions.ipynb` and `11.1_cross_session_eegnet.ipynb` form a revised branch using the documented GDF channel order and cue events.

- Nine participant-specific models per condition, one seed in the supplied run.
- Training: session T runs 0–4, 240 trials.
- Validation/checkpoint selection: T run 5, 48 trials.
- Test: session E, 288 trials per participant; 2,592 in total.
- Fixed architecture/settings, with training-only augmentation.
- Classification window 0.5–3.5 seconds after the verified cue event; normalization uses the preceding −1 to 0 second baseline.
- Run-wise, offline 1–40 Hz minimum-phase filtering. This is not proof of strict streaming causality, particularly at padded run edges.
- ICA uses only EOG-calibration runs, with components chosen from EOG correlation. The default estimates ICA separately using unlabeled calibration data from T and E; E task trials do not fit ICA. The optional frozen-T alternative is a distinct setting, not a result established here.

| Metric | NoAug | Aug |
|---|---:|---:|
| Balanced accuracy | **0.561 ± 0.136** | **0.561 ± 0.134** |
| Macro F1 | 0.557 ± 0.136 | 0.557 ± 0.133 |
| Cohen's kappa | 0.415 ± 0.182 | 0.415 ± 0.178 |

NoAug participant BA ranges from approximately **0.431 to 0.813**. Four participants improve with augmentation, three decrease and two are unchanged to the displayed precision. Effects are small relative to participant variation. The same participant appearing in T and E is intentional: this assesses session transfer, not a new person.

Artifact flags are audited but not excluded in the supplied run: all 288 trials per session are retained. The audit is not evidence that every retained trial is artifact-free.

## 7. Physiological inspection of frozen cross-session models

`11.2_interpretability_mu_beta.ipynb` revalidates checkpoint predictions, then distinguishes spectral perturbation, channel occlusion and descriptive ERD/ERS. Variant A removes bands from normalized windows; variant B, added after A, removes them from raw windows and re-estimates pre-cue normalization. It is post hoc and must be identified as such.

In NoAug variant B, removing mu (8–12 Hz) changes BA by approximately:

| Participant | BA change |
|---|---:|
| A01 | −0.344 |
| A03 | −0.295 |
| A08 | −0.101 |
| A09 | −0.365 |

These four participants have larger drops than the specified mu control bands, with paired within-participant trial-bootstrap intervals excluding zero. This descriptive finding does not prove selective decoding of ERD dynamics: band removal can produce a distribution shift and concentration of predictions in one class. Beta dependence is not generally established; the marked exception in variant B is A08.

Occluding C3/Cz/C4 is unusually damaging relative to random equal-size triples mainly in A09, rather than consistently across participants. ERD/ERS topographies characterize signal changes and are not evidence that the model specifically uses those changes. Bootstrap intervals condition on one participant/session and do not represent uncertainty across people, sessions or model seeds.


