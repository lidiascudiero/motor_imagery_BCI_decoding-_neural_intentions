#  Motor Imagery BCI: Decoding Neural Intentions

This project implements a complete **Brain-Computer Interface (BCI)** pipeline to decode motor imagery (left hand vs. right hand) from EEG signals. Moving beyond simple classification, the focus is on **advanced signal preprocessing** and **artifact rejection** to ensure high-fidelity neural decoding.
---

##  Project Overview
Decoding neural signals is a challenge of **Signal-to-Noise Ratio (SNR)**. EEG data is notoriously noisy, contaminated by ocular (EOG) and muscular (EMG) artifacts. This project applies a rigorous **neuro-engineering approach** to transform raw, noisy brainwaves into actionable commands.

- **Dataset:** BCI Competition IV-2a (9 subjects, Motor Imagery).
- **Primary Architecture:** CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis).
- **Experimental Feature:** Deep Learning (EEGNet/Gaussian variations) for non-linear feature extraction.

---

## The "Neuro-Data" Angle 
> **Author's Note:** Leveraging my background in **Cognitive Neuroscience**, I approached this EEG dataset not just as a stochastic time series, but as a window into neural dynamics. My expertise in biological signals allows me to bridge the gap between physiological integrity and algorithmic precision. In this project, I focused on how specific neural signatures—such as **Mu/Beta desynchronization (ERD)**—translate into actionable digital commands, ensuring that the model’s "attention" remains neurobiologically grounded over the motor cortex rather than fitting to stochastic noise.

---
##  The Pipeline: From Brain to Command

### 1. Advanced Signal Cleaning (The "Neuro" Edge)
The most critical phase. Without high-quality data, even the best model fails.
* **Temporal Filtering:** Band-pass filtering (8-30 Hz) to isolate **Mu and Beta rhythms**.
* **Artifact Mitigation (ICA):** Applied **Independent Component Analysis** to isolate and remove blinks (EOG) without destroying underlying neural signals.
* **Re-referencing:** Applied Common Average Reference (CAR) to increase the SNR.

### 2. Feature Extraction & Decoding
* **Spatial Filtering (CSP):** Implemented **Common Spatial Patterns** to maximize variance between classes, transforming high-dimensional EEG space into an optimized discriminative space.
* **Classification:** Used **LDA** for its robustness and low latency, making it ideal for real-time applications.

### 3. Real-Time Simulation & Demo
I developed a **BCI Simulation environment** to test the model's performance in a pseudo-real-time scenario, simulating how a system reacts to neural commands.

##  Repository Structure: The Full Pipeline

1.  **`1.exploratory_analysis.ipynb`**: Preliminary inspection, PSD (Power Spectral Density) analysis, and signal visualization.
2.  **`2.preprocessing_signal_cleaned.ipynb`**: Core preprocessing logic: filtering, CAR, and ICA artifact rejection.
3.  **`3.mi_decoding_CSP_LDA.ipynb`**: The main decoding engine using CSP spatial filters and LDA classification.
4.  **`4.0.bci_simulation.ipynb`**: Jupyter-based simulation of a BCI session to validate the model's response.
5.  **`4.1.bci_demo.py`**: Python script for a modular BCI demonstration.
6.  **`5.0.preprocessing_signal_dl.ipynb`**: Specific preprocessing pipeline tailored for Deep Learning input shapes.
7.  **`5.deep_learning_guassian_class.ipynb`**: Implementation of Deep Learning models (Gaussian-based classifiers) for neural decoding.
8.  **`6.bci_simulation.ipynb`**: Advanced simulation notebook for performance benchmarking.
9.  **`7.bci_demo.py`**: Refined standalone demo script for system-level testing.
    
## 4. Interactive BCI Demos 

Experience the BCI decoding pipeline in real-time through two dedicated Streamlit applications. These dashboards simulate live BCI sessions, visualizing how neural signals are transformed into motor commands.

---

###  Binary Decoding (2-Class Simulation)
*Focus: Traditional CSP + LDA pipeline for Left vs. Right hand imagery.*

 [**Access Binary BCI Dashboard**](https://motorimagerybcidecoding-neuralintentions-cthqwdblnysiclnn3qkyv.streamlit.app/)

* **Real-Time Simulation:** Observe the decoding process as the system processes EEG epochs window-by-window.
* **Spatial Feature Visualization:** View the **CSP Topomaps** to verify the model is correctly targeting the motor cortex (C3/C4).
* **Signal Integrity:** Inspect the impact of Mu/Beta band-pass filtering (8-30 Hz) on the raw neural signal.

---

###  Multi-Class Decoding (4-Class Advanced)
*Focus: Deep Learning (EEGNet) for Left Hand, Right Hand, Feet, and Tongue imagery.*

 [**Access Multi-Class BCI Dashboard**](https://motorimagerybcidecoding-neuralintentions-cthqwdblnysiclnn3qkyv.streamlit.app/)

* **Deep Learning Inference:** Real-time predictions using a pre-trained **EEGNet** model (optimized for 4-class discrimination).
* **Probability Distribution:** A dynamic bar chart visualizes the model's confidence across all four classes in real-time.
* **Smart Streamer Logic:** Connects to a simulated LSL stream with automatic exit conditions, mimicking a clinical recording session.

---

> **Note:** Both demos utilize pre-processed `.fif` files (Subject A07T) to ensure optimal performance and allow for a focus on the real-time decoding logic and UI feedback. For architectural clarity, the repository is organized into independent directories, each containing its own specific preprocessing pipeline (8-30 Hz for Binary vs. 1-40 Hz for Multi-Class) and dedicated environment requirements.

##  Tech Stack

* **Language:** Python
* **Neuro-Signal Processing:** `MNE-Python`
* **Machine Learning:** `Scikit-learn` (CSP + LDA Pipeline)
* **Deep Learning:** `TensorFlow/Keras` (EEGNet Architecture)
* **Real-Time Data Streaming:** `PyLSL` (Lab Streaming Layer)
* **Deployment and UI:** `Streamlit` (Cloud Hosting and Dashboarding)
* **Visualization:** `Matplotlib`, `Seaborn`, `Plotly`
