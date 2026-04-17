""""The streamer is designed with a sample-limit exit condition. T
his ensures that the simulation mimics a standard clinical recording session, preventing unnecessary 
CPU overhead once the subject's dataset is fully processed."""


import streamlit as st
import numpy as np
import tensorflow as tf
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet
import pandas as pd
import threading
import mne
import time
import os

# --- Configuration ---
MODEL_PATH = 'final_eegnet_bci_model.keras'
CHANNELS = 22
WINDOW_SIZE = 751
SFREQ = 250.0
CLASS_NAMES = ['Left Hand', 'Right Hand', 'Foot', 'Tongue']
CLEAN_DATA_DIR = os.path.join('.', 'cleaned_data_1_40')

st.set_page_config(page_title="BCI Real-Time Decoder", layout="wide")

# --- 1. SMART STREAMER LOGIC ---
def start_smart_streamer(subject_id='A07T'):
    """Simulates an EEG headset with an automatic stop when data is exhausted."""
    def streamer_loop():
        file_path = os.path.join(CLEAN_DATA_DIR, f'{subject_id}_clean-raw_1_40.fif')
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        raw.pick(['eeg'])
        data = raw.get_data()
        
        info = StreamInfo('MockEEG', 'EEG', CHANNELS, SFREQ, 'float32', 'bci_uid_001')
        outlet = StreamOutlet(info)
        
        # The 'sample-limit exit condition' mentioned in your comment
        for sample_idx in range(data.shape[1]):
            # Check if thread should be interrupted
            if not getattr(threading.current_thread(), "do_run", True):
                break
                
            sample = data[:, sample_idx].tolist()
            outlet.push_sample(sample)
            time.sleep(1.0 / SFREQ)
            
        print("[STREAMER] Data exhausted. Session finished.")

    t = threading.Thread(target=streamer_loop)
    t.do_run = True
    t.start()
    return t

@st.cache_resource
def load_bci_model():
    """Loads the pre-trained EEGNet model once."""
    return tf.keras.models.load_model(MODEL_PATH)

# --- 2. MAIN STREAMLIT APP ---
def run_streamlit_bci():
    st.title("Real-Time Motor Imagery Decoding")
    st.markdown("""
    This dashboard visualizes neural predictions from an **EEGNet** model (Accuracy: 63.4%). 
    It connects to a simulated LSL stream and processes data in 3-second windows.
    """)

    # Sidebar for Status
    st.sidebar.header("System Status")
    status_placeholder = st.sidebar.empty()
    
    # Start the streamer automatically if not already running
    if 'streamer_active' not in st.session_state:
        st.session_state.streamer_thread = start_smart_streamer('A07T')
        st.session_state.streamer_active = True
        status_placeholder.info("Starting Smart Streamer...")
        time.sleep(2) # Give LSL time to initialize

    # Main UI Components
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Detected Intent")
        prediction_text = st.empty()
        confidence_metric = st.empty()

    with col2:
        st.subheader("Class Probabilities")
        chart_placeholder = st.empty()

    # Model and LSL Setup
    model = load_bci_model()
    streams = resolve_byprop('name', 'MockEEG', timeout=5)
    
    if not streams:
        status_placeholder.error("LSL Stream 'MockEEG' not found.")
        return

    inlet = StreamInlet(streams[0])
    status_placeholder.success("Connected to MockEEG")
    
    buffer = np.zeros((CHANNELS, WINDOW_SIZE))
    
    # Real-time Loop
    try:
        while True:
            # pull_sample with timeout to detect when the smart streamer stops
            sample, timestamp = inlet.pull_sample(timeout=2.0)
            
            if sample is None:
                status_placeholder.error("Session Finished: Stream Closed.")
                break

            buffer = np.roll(buffer, -1, axis=1)
            buffer[:, -1] = sample

            # Inference every 0.5 seconds (125 samples at 250Hz)
            if int(timestamp * SFREQ) % 125 == 0:
                # Pre-processing: Z-score standardization per trial
                inp = (buffer - np.mean(buffer, axis=-1, keepdims=True)) / np.std(buffer, axis=-1, keepdims=True)
                inp = inp.reshape(1, CHANNELS, WINDOW_SIZE, 1)
                
                # Model Prediction
                probs = model.predict(inp, verbose=0)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]

                # Update Charts
                chart_data = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': probs})
                chart_placeholder.bar_chart(chart_data, x='Class', y='Probability', color="#0072B2")

                # Update Prediction Display with 60% threshold
                if confidence > 0.60:
                    prediction_text.header(f"ACTION: {CLASS_NAMES[pred_idx]}")
                    confidence_metric.metric("Confidence", f"{confidence:.1%}")
                else:
                    prediction_text.header("ACTION: Neutral")
                    confidence_metric.metric("Confidence", f"{confidence:.1%}", delta_color="off")
    except Exception as e:
        st.error(f"Error during decoding: {e}")

if __name__ == "__main__":
    run_streamlit_bci()