# Methods and results by evaluation setting

This document separates reported experiments by task, evaluation unit and preprocessing. Values come from embedded outputs in the supplied notebooks; they have not been independently reproduced by rerunning all training jobs.

## 1. Binary CSP–LDA: resolving the 77% versus 65.7% figures

Notebook `03_mi_decoding_CSP_LDA_.ipynb` evaluates left versus right imagery with ten stratified ShuffleSplit iterations per participant, using an 80/20 training/test split. CSP and LDA are in a scikit-learn pipeline, so CSP is fitted separately inside each training split.

| Reported value | Actual scope |
|---|---|
| Mean accuracy 0.657; between-subject SD 0.084 | Mean of the nine participant-specific CV estimates |
| Mean accuracy 0.769 | A01T only, across ten repeated splits |
| Aggregated accuracy 0.769; BA 0.769 | A01T only: 290 test-prediction records from repeated splits |
| Mean accuracy 0.779 | A07T only, across ten repeated splits |

Each participant has 144 unique binary trials. A repeated-split aggregate can contain the same trial more than once: A01's 290 predictions are not 290 independent trials. The approximately 77% result should not be described as overall performance across nine participants or as demo inference accuracy.

Similar left/right class metrics describe that class comparison only. They do not demonstrate fairness across participants, demographics or acquisition conditions.

### Preprocessing boundary

The original preprocessing notebooks fit ICA on the participant's full recording before trial-level CV or participant-based model evaluation. This is an offline, unlabeled preprocessing step that uses data beyond the model-training split. Keeping CSP in a pipeline does not establish that the entire signal-processing workflow is strictly inductive.

Saved CSP logs report a 25-channel input rank. The dataset contains 22 EEG plus three EOG channels. Audit and explicitly select EEG inputs before presenting this result as an EEG-only CSP benchmark. The early manually assigned channel list differs from the official list used in notebook 11.0, so historical scalp locations need correction or provenance checks before anatomical interpretation. No corrected numerical result is supplied here.

## 2. Fixed-participant holdout EEGNet

`08.0_deep_learning_guassian_class_sub_holdout.ipynb` uses five training, two validation and two test participants on the four-class task. Test participants are A02T and A09T: 576 trials in total, 144 per class.

- Overall test accuracy: **0.470**.
- A02T accuracy: **0.382**.
- A09T accuracy: **0.559**.

The small test cohort makes this result sensitive to participant assignment. The three-seed band/augmentation ablation in that notebook reports validation-subject results, not additional independent test replications. Its 1–40 Hz NoAug validation accuracy is 0.531 ± 0.025; augmentation is 0.534 ± 0.024; 8–30 Hz is 0.350 ± 0.014.

The earlier pooled-random-split value near 63% is retained as development history rather than a headline result: its exact originating training artifact and split were not established from the supplied training notebooks. In particular, the four-class demo loads a checkpoint whose filename is also used by the later fixed-holdout experiment, so the filename alone does not identify training membership or performance.

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

Gaussian noise has sigma 0.05 in z-scored units, applied to half the training trials on average. It is not applied to validation or test data. The experiment matches architecture, seeds, shared hyperparameters and dataset size. However, final epoch counts can differ by condition, so final optimization-update counts are not necessarily identical. Avoid an unqualified claim of exactly compute-matched final fits.

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

RA-full is a transductive batch setting; it must not be described as a model operating on the first target trial with no target data. Depending on the implementation, calibration trials can overlap evaluation trials when their labels are withheld; this still changes target-access assumptions and should be disclosed.

The Riemannian notebooks import historical neural result tables with EEGNet BA **0.532/0.540**, not the current nested-LOSO **0.553/0.553**. Do not label that imported comparison as a benchmark against the latest nested run. One adaptation output explicitly skips a missing attention-model CSV.

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

## 8. Demonstration boundaries

The supplied `4.1.bci_demo (1).py` fits CSP for visualization but contains no LDA inference. It displays arrows from `epochs.events` and generates confidence with `np.random.uniform(70, 78)`. Describe this as a label-driven interface simulation. The similarly named original `.py` in Downloads should not be assumed to differ without checking it.

`04_bci_simulation.ipynb` fits CSP–LDA on A07 trials and reuses that participant's trials for illustration; this is not a held-out evaluation.

`7.bci_demo.py` loads a Keras model and invokes `predict` on simulated LSL windows. It replays preprocessed A07T data, not fresh prospective EEG. The supplied script also contains duplicated application code, which should be consolidated before presenting a canonical demo entry point.

The pretrained model's training membership and input preprocessing must be established from the checkpoint's own provenance before claiming new-user performance or assigning the historical 63.4% score to the deployed model.
