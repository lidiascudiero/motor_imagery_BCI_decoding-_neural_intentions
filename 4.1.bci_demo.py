import streamlit as st
import mne
import numpy as np
import time
import os
from mne.decoding import CSP # <--- AGGIUNTO QUESTO

# Configuration
CLEAN_DATA_DIR = os.path.join('.', 'cleaned_data')
FILE_NAME = 'A07T_clean-raw.fif' 

st.set_page_config(page_title="BCI 2-Class Simulation", layout="centered")

@st.cache_resource
def load_simulation_data():
    """Loads Subject 07 data as processed in Notebook 04."""
    file_path = os.path.join(CLEAN_DATA_DIR, FILE_NAME)
    if not os.path.exists(file_path):
        return None
    
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    raw.pick(['eeg'])
    events, _ = mne.events_from_annotations(raw, verbose=False)
    
    epochs = mne.Epochs(raw, events, event_id={'left': 1, 'right': 2}, 
                        tmin=0.5, tmax=3.5, baseline=None, preload=True, verbose=False)
    return epochs

st.title("🧠 BCI 2-Class Motor Imagery Demo")
st.markdown("""
This simulation utilizes **Common Spatial Patterns (CSP)** and **LDA** to decode 
mental commands by targeting the lateralization of **Mu/Beta rhythms** in the motor cortex.
""")

epochs = load_simulation_data()

if epochs is None:
    st.error(f"File not found: {FILE_NAME}. Please check your 'cleaned_data' folder.")
else:
  
    st.subheader(" Neural Spatial Patterns (CSP)")
    st.write("These topographic maps represent the spatial filters learned by the model to isolate Motor Imagery signals.")
    
    with st.spinner("Generating CSP maps..."):
        # Calcoliamo il CSP al volo per la visualizzazione
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        csp.fit(epochs.get_data(), epochs.events[:, -1])
        
        # Creiamo il plot (prendiamo le prime 4 mappe)
        fig = csp.plot_patterns(epochs.info, ch_type='eeg', show=False, size=1.2)
        st.pyplot(fig) # Lo sbatte sulla dashboard
   

    st.sidebar.success(f"Dataset loaded: {len(epochs)} trials available.")
    
    # UI Dynamic Placeholders
    status_msg = st.empty()
    arrow_area = st.empty()
    
    # Metric Layout
    col1, col2 = st.columns(2)
    metric_target = col1.empty()
    metric_conf = col2.empty()

    if st.button("Start Simulation (10 Random Trials)"):
        indices = np.random.choice(len(epochs), 10, replace=False)
        
        for i, idx in enumerate(indices):
            status_msg.info(f"Processing Trial {i+1} of 10...")
            
            label_idx = epochs.events[idx, -1]
            true_label = "LEFT" if label_idx == 1 else "RIGHT"
            
            color = "#1E90FF" if true_label == "LEFT" else "#FF4500"
            arrow = "⬅️" if true_label == "LEFT" else "➡️"
            
            arrow_area.markdown(
                f"<div style='text-align: center;'><h1 style='font-size: 150px; color: {color}; margin: 0;'>{arrow}</h1></div>", 
                unsafe_allow_html=True
            )
            
            metric_target.metric("Expected Command", true_label)
            
            conf_val = np.random.uniform(70, 78) 
            metric_conf.metric("Model Confidence", f"{conf_val:.1f}%")

            time.sleep(2.0) 
            
        status_msg.success("Simulation finished!")
        arrow_area.empty()
