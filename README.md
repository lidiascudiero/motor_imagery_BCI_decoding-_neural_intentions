#  Motor Imagery BCI: Decoding Neural Intentions

This project implements a complete **Brain-Computer Interface (BCI)** pipeline to decode motor imagery (left hand vs. right hand) from EEG signals. Moving beyond simple classification, the focus is on **advanced signal preprocessing** and **artifact rejection** to ensure high-fidelity neural decoding.

##  Project Overview
Decoding neural signals is a challenge of **Signal-to-Noise Ratio (SNR)**. EEG data is notoriously noisy, contaminated by ocular (EOG) and muscular (EMG) artifacts. This project applies a rigorous **neuro-engineering approach** to transform raw, noisy brainwaves into actionable commands.

- **Dataset:** BCI Competition IV-2a (9 subjects, Motor Imagery).
- **Primary Architecture:** CSP (Common Spatial Patterns) + LDA (Linear Discriminant Analysis).
- **Experimental Feature:** Deep Learning (EEGNet/Gaussian variations) for non-linear feature extraction.

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

## 4. Interactive Live Demo
Experience the BCI decoding pipeline in real-time through the dedicated Streamlit application. This dashboard allows you to simulate a BCI session and visualize how neural signals are transformed into motor commands.

 [**Access the Live BCI Dashboard**]([https://motorimagerybcidecoding-neuralintentions-zrq3ni6ifn5tzybufphdm.streamlit.app/](https://motorimagerybcidecoding-neuralintentions-zrq3ni6ifn5tzybufphdm.streamlit.app/)

### What you can explore:
* **Real-Time Simulation:** Observe the decoding process as the system processes EEG epochs from the BCI Competition IV-2a dataset.
* **Spatial Feature Visualization:** View the **CSP Topomaps** generated for each subject to verify that the model is correctly targeting the motor cortex (C3/C4).
* **Performance Analytics:** Monitor classification confidence and accuracy during the simulated "online" session.
* **Signal Integrity:** Inspect the impact of Mu/Beta band-pass filtering (8-30 Hz) and artifact rejection on the raw neural signal.

> **Note:** The demo utilizes pre-processed `.fif` files from the `cleaned_data` folder to ensure optimal performance and focus on the decoding logic.
##  Tech Stack
* **Language:** Python
* **Neuro-Signal Processing:** `MNE-Python`
* **Machine Learning:** `Scikit-learn`
* **Deep Learning:** `TensorFlow/Keras`
* **Visualization:** `Matplotlib`, `Seaborn`
