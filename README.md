#  Motor Imagery BCI: Decoding Neural Intentions

## Overview

This project explores motor imagery (MI) EEG decoding using classical machine learning, deep learning, and Riemannian geometry. It investigates both **cross-subject generalization** and **within-subject cross-session robustness** using the BCI Competition IV 2a dataset.

The pipeline includes signal preprocessing, feature extraction, model selection, data augmentation, subject-wise validation, cross-session evaluation, and post-hoc model interpretability.

The main objectives are to:

- Compare classical CSP-based decoding with neural network approaches.
- Evaluate generalization to unseen participants using subject-wise splits and Leave-One-Subject-Out (LOSO) validation.
- Investigate Gaussian noise augmentation as a potential regularization strategy.
- Assess covariance-based Riemannian classifiers and unsupervised target-subject adaptation.
- Evaluate EEGNet robustness across recording sessions.
- Investigate whether model predictions are consistent with established sensorimotor EEG patterns.

## Dataset

The experiments use the **BCI Competition IV Dataset 2a**, which contains EEG recordings from nine participants performing four motor imagery tasks:

- Left hand
- Right hand
- Both feet
- Tongue

The dataset supports both multiclass motor imagery decoding and binary classification experiments.

## Neurophysiological Motivation: From EEG Signals to Motor Intentions

> **Author's note**
>
> Leveraging my background in Cognitive Neuroscience, I approached EEG decoding not simply as a classification problem on stochastic time series, but as an opportunity to investigate the relationship between neural dynamics and algorithmic predictions.
>
> The project explores whether sensorimotor signatures associated with motor imagery—particularly **mu/beta event-related desynchronization (ERD)**—can support reliable digital commands through a systematic decoding pipeline.
>
> Beyond classification accuracy, I inspected CSP spatial patterns and scalp topographies to assess their qualitative consistency with established sensorimotor physiology, including regions around C3 and C4 commonly associated with hand motor imagery.
>
> These visualizations provide evidence of **physiological plausibility**, not proof of cortical source localization. Scalp EEG topographies are affected by volume conduction and do not uniquely identify neural generators without dedicated source-reconstruction methods.
>
> This perspective informed the investigation of filtering, artifact mitigation, spatial feature extraction, and model evaluation, with the aim of assessing whether decoding performance was consistent with task-relevant sensorimotor activity rather than relying primarily on noise, artifacts, or dataset-specific correlations.

## Interactive BCI Demonstrations

Two Streamlit applications demonstrate the decoding pipeline through interactive simulations of EEG-based motor imagery classification.

### Binary Motor Imagery Decoding

**Task:** Left-hand vs. right-hand motor imagery.

**Architecture:** Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA).

[**Launch the Binary BCI Dashboard**](https://lidiascudiero-demo-2-class-bci.hf.space)

Features include:

- **Window-by-window decoding simulation:** observe the classification process across EEG epochs.
- **Performance evaluation:** the reported aggregated test accuracy is approximately 77%. A separate summary reports a mean within-subject accuracy of 65.7% ± 8.4% across nine participants.
- **Class-wise metrics:** the reported balanced accuracy is 0.769, with F1-scores of 0.767 for left-hand imagery and 0.771 for right-hand imagery.
- **CSP topographies:** inspect spatial patterns and their qualitative relationship with sensorimotor scalp regions.
- **Mu/beta filtering:** explore the effect of 8–30 Hz band-pass filtering on the EEG signal.

The reported class-wise metrics are similar, suggesting broadly comparable performance for the two classes in that evaluation. They do not, by themselves, establish that the model is unbiased across participants or recording conditions.

### Four-Class Motor Imagery Decoding

**Task:** Left hand, right hand, both feet, and tongue imagery.

**Architecture:** EEGNet, a compact convolutional neural network for EEG decoding.

[**Launch the Multi-Class BCI Dashboard**](https://lidiascudiero-demo-4-class-bci.hf.space)

Features include:

- **EEGNet inference:** generate predictions using a pretrained four-class model.
- **Reported performance:** 63.4% accuracy in the demonstrated setup, compared with a nominal four-class chance level of 25%.
- **Class probability visualization:** inspect the model's predicted probability distribution across the four motor imagery classes.
- **LSL streaming simulation:** connect to a simulated Lab Streaming Layer (LSL) stream with automatic exit conditions to demonstrate the flow of an EEG decoding session.

The 63.4% result describes the specific demo evaluation and should not be directly compared with the LOSO or cross-session results without matching the evaluation protocols.

### Demo Data and Implementation Notes

Both demonstrations use preprocessed `.fif` recordings from participant A07, session `T`, to focus on the decoding workflow and user-interface feedback.

The binary and multiclass applications use separate preprocessing pipelines:

| Component | Binary demo | Multiclass demo |
|---|---|---|
| Task | Left vs. right hand | Left hand, right hand, feet, tongue |
| Classifier | CSP + LDA | EEGNet |
| Reported frequency range | 8–30 Hz | 1–40 Hz |
| Input data | Preprocessed `.fif` recordings | Preprocessed `.fif` recordings |
| Application | Streamlit | Streamlit |

Each application has its own preprocessing configuration and environment requirements. The demonstrations illustrate the inference workflow; they should not be interpreted as evidence of a fully validated, online BCI operating on newly acquired EEG unless that capability is separately evaluated.


## Experimental Pipeline

### 1. Classical Baseline: CSP + LDA

Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA) provide a classical baseline for motor imagery classification.

CSP extracts spatial filters that emphasize differences in class-specific signal variance, while LDA performs the final classification.

Reported results include:

- **Mean within-subject accuracy:** 65.7% ± 8.4% across nine participants.
- **Separate reported test result:** approximately 77% aggregated accuracy, with a reported top-performing subject result of 77.9%.

These figures refer to different reported summaries and should not be interpreted as the same evaluation metric or protocol. Exact split definitions and aggregation details should be documented before making direct comparisons with other models.

### 2. Initial Cross-Subject Deep Learning Experiments

Initial experiments compared two neural architectures:

- **EEGNet**, a compact convolutional neural network designed for EEG decoding.
- **AttentionBaseNet**, a CNN incorporating channel-attention mechanisms, including Squeeze-and-Excitation modules.

A fixed subject-wise split was used:

- 5 subjects for training
- 2 subjects for validation
- 2 subjects for testing

Grid search for AttentionBaseNet identified the following configuration:

| Hyperparameter | Selected value |
|---|---:|
| Learning rate | 0.001 |
| Dropout | 0.5 |
| Batch size | 32 |

This protocol evaluates generalization to held-out participants. However, because the test set contains only two subjects, the result may be sensitive to which participants are assigned to the test set.

### 3. Gaussian Noise Augmentation

Training-only Gaussian noise augmentation was evaluated for EEGNet and AttentionBaseNet.

The augmentation adds Gaussian noise with standard deviation \(\sigma = 0.05\) to training data. The aim is to improve robustness and reduce overfitting.

Across the reported experiments, augmented and non-augmented models showed very similar aggregate performance, with differences small relative to the variability across subjects.

In the nested LOSO EEGNet experiment:

| Metric | NoAug | Aug |
|---|---:|---:|
| Accuracy | 55.3% ± 9.0% | 55.3% ± 8.5% |
| Balanced accuracy | 55.3% ± 9.0% | 55.3% ± 8.5% |
| Macro-F1 | 54.0% ± 9.7% | 54.2% ± 8.8% |
| Cohen's kappa | 0.404 ± 0.120 | 0.404 ± 0.113 |

The results do not demonstrate a meaningful average improvement from this augmentation setting. Small differences between individual subjects may reflect optimization variability or subject-specific effects. Establishing equivalence would require a dedicated statistical analysis and an appropriately defined equivalence margin.

### 4. Cross-Subject Evaluation: Leave-One-Subject-Out (LOSO)

To evaluate generalization across participants more systematically, the pipeline includes a nine-fold LOSO evaluation.

In each outer fold, one participant is held out for testing while the remaining participants are used for training and model selection according to the experiment's validation protocol.

This evaluation is more informative about unseen-subject generalization than a random trial-level split in which trials from the same participant can appear in both training and testing.

The nested LOSO EEGNet results reported above indicate that four-class decoding across unseen participants remains challenging, with substantial differences in performance across subjects.

**Protocol note:** the exact validation procedure should be described per experiment. A fixed validation subject in each outer fold is not equivalent to a full inner cross-validation loop, even when both use an outer LOSO test split.

### 5. Riemannian Geometry Baselines

Covariance-based Riemannian methods were evaluated as an alternative to conventional feature extraction and deep learning.

The pipeline includes:

- **MDM (Minimum Distance to Mean):** classifies samples according to their Riemannian distance from class covariance means.
- **Tangent-space logistic regression:** maps covariance matrices to a tangent space and applies logistic regression.
- **Riemannian Alignment (RA):** aligns covariance representations across subjects.
- **Tangent-space updating (`tsupdate`):** updates the target-domain tangent-space reference using target data.

Under inductive LOSO evaluation, MDM and tangent-space logistic regression with `none` and `zscore` normalization achieved balanced accuracy of approximately **0.26–0.28**, close to the four-class chance level of 0.25.

Transductive adaptation improved reported balanced accuracy to approximately **0.31–0.34** using RA-full, RA-calib48, and tangent-space updating.

These findings suggest that covariance-based methods may benefit from target-subject alignment when inter-subject variability is substantial. However, the adapted results should not be directly compared with strictly inductive results without clearly identifying the target data used during adaptation.

**Evaluation distinction:**

- **Inductive LOSO:** the held-out subject is not used to adapt the model or reference representation.
- **Transductive adaptation:** unlabeled target-subject data may be used to estimate alignment or update a representation.
- **Supervised adaptation:** uses target-subject labels and must be reported separately.

Transductive adaptation does not necessarily involve label leakage, but it changes the assumptions about target data availability and deployment.

### 6. Within-Subject Cross-Session Generalization with EEGNet

A separate experiment investigates temporal robustness by training and validating on one recording session and testing on a distinct session for the same participant.

**Protocol**

- Training: session `T`, runs 0–4.
- Validation and early stopping: session `T`, run 5.
- Testing: session `E`.
- Preprocessing: run-wise 1–40 Hz minimum-phase filtering and pre-cue baseline normalization over −1.0 to 0 seconds relative to the cue.
- ICA: calibration-only estimation, as documented for this experiment.

The purpose is to measure whether a model trained on one session can decode motor imagery in a separate session without using the test session's labels for model fitting.

**Reported results across nine participants**

| Metric | NoAug | Aug |
|---|---:|---:|
| Balanced accuracy | 56.1% ± 13.4% | 56.1% ± 13.4% |
| Macro-F1 | 55.7% ± 13.6% | 55.7% ± 13.3% |
| Cohen's kappa | 0.415 ± 0.182 | 0.415 ± 0.178 |

The reported cross-session results indicate moderate decoding performance with considerable inter-subject variability. Gaussian augmentation did not produce a clear aggregate improvement.

*Reporting note:* the experiment summary reports balanced accuracy of 56.1% ± 13.4% for both conditions. Confirm the saved result tables before publication if the underlying notebook contains a different standard deviation for either condition.

### 7. Model Interpretability

Post-hoc interpretability analyses were performed on frozen cross-session EEGNet checkpoints.

The analyses include:

- **Frequency-band perturbation:** evaluating changes in predictions after removing mu, beta, or combined mu and beta activity.
- **Channel occlusion:** investigating how predictions change when individual EEG channels are masked or perturbed.
- **Class confusion analysis:** identifying which motor imagery classes are most frequently confused.

These methods can help assess whether model predictions are sensitive to physiologically relevant sensorimotor rhythms and scalp channels.

Interpretation must remain cautious: perturbation-based analyses describe model sensitivity under a particular intervention, not causal brain mechanisms or definitive neural source localization.

## Results Summary

The following table summarizes the main experiments. Results should be interpreted within their respective evaluation protocols.

| Experiment | Evaluation protocol | Reported result | Main observation |
|---|---|---|---|
| CSP + LDA | Within-subject binary decoding; exact split to document | 65.7% ± 8.4% mean accuracy; separate reported result around 77% | Classical baseline; summaries require protocol clarification |
| Pooled deep learning | Random 80/20 trial split across pooled subjects | Approximately 63% accuracy | Trials from the same subjects can appear in train and test |
| Subject-wise deep learning | 5 training, 2 validation, 2 test subjects | Approximately 47% test accuracy | Evaluates held-out subjects, but only two test subjects |
| EEGNet nested LOSO | Nine outer subject folds | 55.3% mean accuracy without and with augmentation | More systematic assessment of unseen-subject generalization |
| Riemannian, inductive LOSO | Held-out subject without target adaptation | Approximately 0.26–0.28 balanced accuracy | Close to four-class chance level |
| Riemannian, transductive adaptation | Target covariance alignment or tangent-space updating | Approximately 0.31–0.34 balanced accuracy | Adaptation improves the reported results under different assumptions |
| EEGNet cross-session | Train/validate on session T, test on session E | 56.1% balanced accuracy; 55.7% Macro-F1 | Moderate within-subject cross-session performance |

**Important:** these experiments differ in task, split strategy, training data, adaptation assumptions, and evaluation metrics. They should not be ranked as if they were measured under a single benchmark protocol.

## Technology Stack

### Programming and Data Analysis

- **Python** — main programming language.
- **NumPy** — numerical computation and array manipulation.
- **Pandas** — tabular data processing.
- **SciPy** — signal processing and statistical utilities.

### EEG Processing and Classical Machine Learning

- **MNE-Python** — EEG data loading, preprocessing, epoching, filtering, and signal analysis.
- **scikit-learn** — LDA, logistic regression, preprocessing, model selection, and evaluation metrics.
- **pyRiemann** — covariance-based classification, Riemannian geometry, tangent-space representations, and related domain-alignment workflows.

### Deep Learning

- **TensorFlow / Keras** — EEGNet implementation and neural network training.
- **PyTorch** — tensor computation and neural network development.
- **Braindecode** — EEG-focused deep learning utilities and model components, used in the AttentionBaseNet workflow.

### Visualization and Applications

- **Matplotlib** and **Seaborn** — scientific plots and statistical visualizations.
- **Plotly** — interactive visualizations.
- **Streamlit** — interactive EEG decoding demonstrations.
- **PyLSL** — Lab Streaming Layer integration for real-time or streaming EEG workflows.

## Reproducibility and Evaluation Considerations

For reproducible and scientifically interpretable comparisons:

1. Record the exact participant IDs assigned to training, validation, and test sets.
2. Report the number of trials per split and per class.
3. Fit normalization, feature extraction, and other learned preprocessing steps using training data only, unless the experiment explicitly studies adaptation.
4. Keep test labels out of model selection and threshold tuning.
5. Report whether target-session data are used for unsupervised calibration, ICA estimation, covariance alignment, or tangent-space updating.
6. Distinguish subject-wise generalization from cross-session generalization.
7. Report per-subject metrics alongside mean and standard deviation.
8. Compare augmentation conditions using paired per-subject results and, where appropriate, repeated-seed experiments.
9. Document random seeds, software versions, and exact evaluation scripts.
10. Avoid direct comparisons between results obtained with different tasks, metrics, or data-splitting protocols.

## Repository Structure

The repository contains notebooks and scripts for exploratory analysis, preprocessing, classical decoding, deep learning, cross-subject evaluation, cross-session experiments, and demonstrations.

Representative components include:

- `exploratory_analysis.ipynb` — exploratory data analysis.
- `preprocessing_signal_cleaned.ipynb` — EEG preprocessing.
- `mi_decoding_CSP_LDA_.ipynb` — CSP + LDA decoding.
- `bci_simulation.ipynb` — BCI simulation workflow.
- `bci_demo.py` — demonstration script.
- `5.deep_learning_guassian_class.ipynb` — pooled trial-level deep learning experiment.
- `8_deep_learning_guassian_class_sub_holdout.ipynb` — subject-wise holdout experiment.
- `8.2_nested_loso.ipynb` — EEGNet LOSO evaluation.
- `cross_session_eegnet.ipynb` — cross-session EEGNet experiment.

Exact filenames and the availability of additional Riemannian or interpretability notebooks should be checked against the current repository.

## Conclusion

This project compares classical spatial filtering, compact neural networks, attention-based architectures, and Riemannian covariance methods for motor imagery EEG decoding.

The experiments highlight three main challenges:

1. **Inter-subject variability:** performance on unseen participants remains difficult, even when using deep learning.
2. **Session variability:** within-subject performance can degrade or vary substantially across separate recording sessions.
3. **Adaptation assumptions:** unsupervised target alignment can improve Riemannian results, but it requires target data and must be evaluated separately from strictly inductive generalization.

Gaussian noise augmentation with \(\sigma = 0.05\) did not yield a clear aggregate improvement in the reported experiments. The results also motivate further investigation into subject adaptation, robust evaluation protocols, and physiologically informed model interpretation.

Overall, the repository provides an experimental framework for studying motor imagery EEG decoding under multiple generalization settings rather than a single, directly comparable leaderboard of model performance.


##  Tech Stack

* **Language:** Python
* **Neuro-Signal Processing:** `MNE-Python`
* **Machine Learning:** `Scikit-learn` (CSP + LDA pipeline),`pyRiemann` (Riemannian geometry)
* **Deep Learning:** `TensorFlow/Keras` (EEGNet architecture), `Braindecode` (AttentionBaseNet)
* **Real-Time Data Streaming:** `PyLSL` (Lab Streaming Layer)
* **Deployment and UI:** `Streamlit` (cloud hpsting and dashboarding)
* **Visualization:** `Matplotlib`, `Seaborn`, `Plotly`
