#  Motor Imagery BCI: Decoding Neural Intentions

This project implements a complete **Brain-Computer Interface (BCI)** pipeline to decode motor imagery (left hand vs. right hand) from EEG signals. Moving beyond simple classification, the focus is on **advanced signal preprocessing** and **artifact rejection** to ensure high-fidelity neural decoding.
---

##  Project Overview
Decoding neural signals is a challenge of **Signal-to-Noise Ratio (SNR)**. EEG data is notoriously noisy, contaminated by ocular (EOG) and muscular (EMG) artifacts. This project applies a rigorous **neuro-engineering approach** to transform raw, noisy brainwaves into actionable commands.

- **Dataset:** BCI Competition IV-2a (9 subjects, Motor Imagery).
- **Primary Architecture:** CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis).
- **Experimental Feature:** Deep Learning (EEGNet/Gaussian variations) for non-linear feature extraction.

---

## The “Neuro-Data” Angle

> **Author’s Note:** Leveraging my background in **Cognitive Neuroscience**, I approached this EEG dataset not simply as a stochastic time series, but as a window into neural dynamics. The aim was to bridge physiological plausibility and algorithmic performance: to investigate whether neural signatures associated with motor imagery—particularly sensorimotor **Mu/Beta event-related desynchronization (ERD)** could be transformed into reliable digital commands through a rigorous decoding pipeline.
>
> Beyond classification accuracy, I examined whether the learned spatial representations were consistent with established neurophysiological expectations. I inspected CSP patterns and topographies to assess their qualitative alignment with sensorimotor scalp regions, including electrodes overlying the C3/C4 areas that are commonly associated with hand motor imagery. This provides evidence of **physiological plausibility**, rather than proof of cortical source localisation: scalp EEG topographies are affected by volume conduction and do not directly identify neural generators without dedicated source-reconstruction methods.
>
> This perspective guided the full pipeline,from artifact mitigation and band-pass filtering to spatial filtering and model evaluation,helping reduce the risk that the classifier relied primarily on noise, non-neural artifacts, or dataset-specific correlations rather than task-relevant sensorimotor patterns.

---
##  The Pipeline: From Brain to Command

### 1. Advanced Signal Cleaning (The "Neuro" Edge)
The most critical phase. Without high-quality data, even the best model fails.
* **Temporal Filtering:** Band-pass filtering (8-30 Hz) to isolate **Mu and Beta rhythms**.
* **Artifact Mitigation (ICA):** Applied **Independent Component Analysis** to isolate and remove blinks (EOG) without destroying underlying neural signals.
* **Re-referencing:** Applied **Common Average Reference (CAR)** to increase the SNR.

### 2. Feature Extraction and decoding
* **Spatial Filtering (CSP):** Implemented **Common Spatial Patterns** to maximize variance between classes, transforming high-dimensional EEG space into an optimized discriminative space.
* **Classification:** Used **LDA** for its robustness and low latency, making it ideal for real-time applications.

### 3. Real-Time Simulation and demo
I developed a **BCI Simulation environment** to test the model's performance in a pseudo-real-time scenario, simulating how a system reacts to neural commands.

##  Repository Structure: The Full Pipeline

1.  **`1.exploratory_analysis.ipynb`**: Preliminary inspection, PSD (Power Spectral Density) analysis, and signal visualization.
2.  **`2.preprocessing_signal_cleaned.ipynb`**: Core preprocessing logic: filtering, CAR, and ICA artifact rejection.
3.  **`3.mi_decoding_CSP_LDA_.ipynb`**: The main decoding engine using CSP spatial filters and LDA classification.
4.  **`4.0.bci_simulation.ipynb`**: Jupyter-based simulation of a BCI session to validate the model's response.
5.  **`4.1.bci_demo.py`**: Python script for a modular BCI demonstration.
6.  **`5.0.preprocessing_signal_dl.ipynb`**: Specific preprocessing pipeline tailored for Deep Learning input shapes.
7.  **`5.deep_learning_guassian_class.ipynb`**: Implementation of Deep Learning models (Gaussian-based classifiers) for neural decoding.
8.  **`5.2.deep_learning_guassian_class_sub.ipynb`**: Implementation of Deep Learning models (Gaussian-based classifiers) for neural decoding with subject wise split.
9.  **`6.bci_simulation.ipynb`**: Advanced simulation notebook for performance benchmarking.
10.  **`7.bci_demo.py`**: Refined standalone demo script for system-level testing.
    
## 4. Interactive BCI Demos 

Experience the BCI decoding pipeline in real-time through two dedicated Streamlit applications. These dashboards simulate live BCI sessions, visualizing how neural signals are transformed into motor commands.

---

###  Binary Decoding (2-Class Simulation)
*Focus: Traditional CSP + LDA pipeline for Left vs. Right hand imagery.*

 [**Access Binary BCI Dashboard**](https://lidiascudiero-demo-2-class-bci.hf.space)

* **Real-Time simulation:** Observe the decoding process as the system processes EEG epochs window-by-window.
* **Performance:** 77% aggregated test accuracy (with top-performing subjects like A07 reaching 77.9%). Across all 9 participants, the model yields a mean within-subject accuracy of 65.7% (±8.4%), highlighting the inherent inter-subject variability in motor imagery BCI applications.
* **Algorithmic fairness:** balanced accuracy matches standard accuracy exactly (0.769), with near-identical F1-scores for Left Hand (0.767) and Right Hand (0.771), confirming the CSP+LDA pipeline has no systematic prediction bias.
* **Spatial feature visualization:** View the **CSP Topomaps** to verify the model is correctly targeting the motor cortex (C3/C4).
* **Signal integrity:** Inspect the impact of Mu/Beta band-pass filtering (8-30 Hz) on the raw neural signal.

---

###  Multi-Class Decoding (4-Class Advanced)
*Focus: Deep Learning (EEGNet) for Left Hand, Right Hand, Feet, and Tongue imagery.*

 [**Access Multi-Class BCI Dashboard**](https://lidiascudiero-demo-4-class-bci.hf.space)

* **Deep Learning inference:** real-time predictions using a pre-trained **EEGNet** model (optimized for 4-class discrimination).
* **Performance:** **63.4% Accuracy** (Chance level: 25%).
* **Probability distribution:** a dynamic bar chart visualizes the model's confidence across all four classes in real-time.
* **Smart Streamer logic:** connects to a simulated LSL stream with automatic exit conditions, mimicking a clinical recording session.

---

> **Note:** Both demos utilize pre-processed `.fif` files (Subject A07T) to ensure optimal performance and allow for a focus on the real-time decoding logic and UI feedback. For architectural clarity, the repository is organized into independent directories, each containing its own specific preprocessing pipeline (8-30 Hz for Binary vs. 1-40 Hz for Multi-Class) and dedicated environment requirements.


### Methodological evolution: pooled evaluation vs. cross-subject generalization

This repository explores two evaluation protocols for four-class motor imagery decoding with deep learning. The distinction is important because performance on previously observed participants does not necessarily translate into reliable predictions for unseen users.

**1. Pooled random split  `5.deep_learning_guassian_class.ipynb`**

The initial experiment pools EEG trials from all nine participants and applies a random 80/20 train-test split using `train_test_split`.

The resulting test set contains trials from participants who are also represented in the training set. The reported accuracy is approximately **63%**.

This protocol evaluates classification performance on held-out trials drawn from a population of participants represented during training. It does not directly measure generalization to entirely unseen individuals, and subject-specific characteristics may contribute to the observed performance.

**2. Subject-wise Evaluation  `5.1_deep_learning_guassian_class_sub.ipynb`**

The revised experiment separates participants before constructing the training, validation, and test datasets:

- **Training:** A01, A03, A05, A06, A08.
- **Validation:** A04, A07.
- **Testing:** A02, A09.

The two test participants are excluded from model training and validation, enabling an evaluation of cross-subject generalization.

The reported test accuracy is **47%**, compared with a nominal chance level of 25% for balanced four-class classification.

**Key takeaway**

 * The difference between the reported accuracies should not be interpreted as a direct comparison of model quality. The experiments use different evaluation protocols and answer different questions.

* The pooled random split measures performance on unseen trials from participants represented during training, whereas the subject-wise protocol evaluates transfer to previously unseen participants.

* This methodological progression highlights an important challenge in motor imagery BCI research: developing decoders that remain effective across individuals rather than relying on subject-specific calibration.

* The cross-subject result is preliminary evidence of performance above nominal chance level, not by itself proof of statistically significant or physiologically interpretable generalization. Further validation across additional participants and repeated subject-wise splits is needed.

##  Tech Stack

* **Language:** Python
* **Neuro-Signal Processing:** `MNE-Python`
* **Machine Learning:** `Scikit-learn` (CSP + LDA Pipeline)
* **Deep Learning:** `TensorFlow/Keras` (EEGNet Architecture)
* **Real-Time Data Streaming:** `PyLSL` (Lab Streaming Layer)
* **Deployment and UI:** `Streamlit` (Cloud Hosting and Dashboarding)
* **Visualization:** `Matplotlib`, `Seaborn`, `Plotly`
