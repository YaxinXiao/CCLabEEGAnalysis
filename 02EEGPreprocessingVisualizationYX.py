"""
EEG Preprocessing Visualization

This script visualizes the effects of preprocessing steps on a selected subject,
including original signal, filtering, bad channel marking, and ICA cleaning.

Steps:
- Load original EDF data
- Apply band-pass filter
- Detect and mark bad channels
- Load ICA-cleaned data
- Plot comparisons between preprocessing stages

Author: Yaxin Xiao
Date: March.2025
"""

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ==== USER PARAMETERS ====
# =========================

# Subject ID to visualize
subjid = '122'

# Input/output paths
edf_folder = 'vs0318'
edf_file = sorted([f for f in os.listdir(edf_folder) if subjid in f and f.endswith('.edf')])[0]
edf_path = os.path.join(edf_folder, edf_file)
fif_path = f'data/Att_VS_ICA_{subjid}.fif.gz'

# EEG cap channel names
ch_names = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
            'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

# Filtering parameters
l_freq = 1.0
h_freq = 63.9

# Bad channel detection threshold
bad_channel_thresh = 3.0

# Time segment for ICA comparison (in seconds)
comparison_start = 260
comparison_duration = 20

# =========================
# ===== LOAD DATA =========
# =========================

# Load original raw data from EDF
raw_orig = mne.io.read_raw_edf(edf_path, preload=True)

# Drop channels not in your cap
ch_df = pd.Series(raw_orig.ch_names)
ch_drop = list(ch_df[~ch_df.isin(ch_names)])
raw_orig.drop_channels(ch_drop)

# Set montage
raw_orig.set_montage('easycap-M1')

# Apply band-pass filter to copy of original
raw_filt = raw_orig.copy().filter(l_freq=l_freq, h_freq=h_freq)

# Function to detect bad channels using z-score
def detect_bad_channels_zscore(raw, threshold=3.0):
    raw_tmp = raw.copy().pick_types(eeg=True)
    data = raw_tmp.get_data()
    stds = np.std(data, axis=1)
    zscores = (stds - np.mean(stds)) / np.std(stds)
    bad_idx = [i for i in range(len(zscores)) if abs(zscores[i]) > threshold]
    bads = [raw_tmp.ch_names[i] for i in bad_idx]
    return bads

# Detect and mark bad channels (for display only)
bads = detect_bad_channels_zscore(raw_filt, threshold=bad_channel_thresh)
raw_filt.info['bads'] = bads
print(f"Detected bad channels: {bads}")

# Load ICA-cleaned data
raw_ica = mne.io.read_raw_fif(fif_path, preload=True)

# =========================
# ======= PLOTS ===========
# =========================

# Plot 1: Original data
raw_orig.plot(title='Original Raw Data', duration=10, n_channels=32)

# Plot 2: Filtered data
raw_filt.plot(title='Band-pass Filtered (1–63.9 Hz)', duration=10, n_channels=32)

# Plot 3: Filtered with bad channels marked
raw_filt.plot(title='Filtered with Bad Channels Marked', duration=10, n_channels=32)

# Plot 4: ICA comparison before vs after
raw_filt.plot(title='Before ICA (Filtered)', start=comparison_start,
              duration=comparison_duration, n_channels=10)
raw_ica.plot(title='After ICA Cleaned', start=comparison_start,
             duration=comparison_duration, n_channels=10)

# Plot 5: ICA-cleaned with bad segments (muscle artifacts) marked
raw_ica.plot(title='After ICA with Bad Segments Marked', duration=10, n_channels=32)

# =========================
# ====== ANNOTATIONS ======
# =========================

# Optional: Print annotation info (e.g., muscle artifact detection)
print("\n=== Annotations in ICA-cleaned data ===")
print(raw_ica.annotations)

for onset, duration, desc in zip(raw_ica.annotations.onset,
                                 raw_ica.annotations.duration,
                                 raw_ica.annotations.description):
    print(f"{desc} at {onset:.2f}s lasting {duration:.2f}s")

# Final layout
plt.tight_layout()
plt.show()
