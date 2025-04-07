"""
Epoch Generation Script for ERP and Time-Frequency Analysis

This script reads ICA-cleaned EEG data (already referenced and filtered),
adds event annotations using vocal stroop trigger markers, and extracts epochs
for ERP or wavelet-based time-frequency analysis.

Steps:
- Load preprocessed EEG data (.fif)
- Load trigger CSV and generate event annotations
- Apply optional re-filtering (depending on ERP/Wavelet)
- Extract epochs and save them for downstream analysis

Author: Yaxin Xiao
Date: March 2025
"""

import os
import re
from glob import glob
import numpy as np
import pandas as pd
import mne

# =========================
# ==== USER PARAMETERS ====
# =========================

# Channel list used in your cap (for reference only, not applied here)
ch_names = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
            'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

# Sampling rate used for latency conversion (if necessary)
sr = 128  # Your latency in CSV is already in seconds, so this is not used.

# File locations
eeg_folder = 'data'
marker_folder = 'vs0318'

# Filtering settings (optional re-filtering before epoching)
filter_params = {
    'erp': dict(l_freq=0.1, h_freq=30),
    'wavelet': dict(l_freq=0.1, h_freq=50)
}

# Epoch timing and baseline correction
# Epoch time settings:
# - ERP: [-0.5s, 2.2s] to capture both early and late components (e.g., P300, LPP)
# - Wavelet: same range to ensure sufficient edge buffer for time-frequency analysis

# Baseline correction:
# - ERP: (-0.2, 0) as baseline
# (None, 0) → full pre-stimulus baseline
# - Wavelet: None → no baseline correction, use power normalization later if needed
epoch_params = {
    'erp': dict(tmin=-0.5, tmax=2.5, baseline=(-0.2, 0)),
    'wavelet': dict(tmin=-0.5, tmax=2.5, baseline=None)
}

# Rejection threshold for ERP (optional)
reject_criteria = dict(eeg=100e-6)  # 100 µV

# =========================
# ==== FILE COLLECTION ====
# =========================

# Gather preprocessed EEG and trigger files
eeg_files = sorted(glob(os.path.join(eeg_folder, '*VS_ICA*fif.gz')))
trig_files = sorted(glob(os.path.join(marker_folder, '*VS*Marker.csv')))

# =========================
# ==== PROCESSING LOOP ====
# =========================

for mode in ['erp', 'wavelet']:
    print(f'\n========== Processing mode: {mode.upper()} ==========')

    for ef in eeg_files:
        # Extract subject ID from filename
        subjid_match = re.findall(r'\d+', ef)
        if not subjid_match:
            print(f"⚠️ Skipping file (no subject ID found): {ef}")
            continue
        subjid = subjid_match[0]

        # Match corresponding trigger file
        tf_matches = [tf for tf in trig_files if subjid in tf]
        if not tf_matches:
            print(f'❌ No trigger file found for subject {subjid}')
            continue
        tf = tf_matches[0]

        # === Load cleaned EEG data ===
        raw = mne.io.read_raw_fif(ef, preload=True)
        raw.apply_proj()  # ensure average reference projection is applied

        # === Optional band-pass filtering for ERP or TFR ===
        raw.filter(**filter_params[mode],
                   h_trans_bandwidth='auto',
                   filter_length='auto',
                   phase='zero')

        # === Load and clean trigger CSV ===
        label = pd.read_csv(tf)

        # Standardize 'type' labels
        label['type'] = (label['type']
                         .str.replace('PM.WAV', 'P')
                         .str.replace('PF.WAV', 'P')
                         .str.replace('NM.WAV', 'N')
                         .str.replace('NF.WAV', 'N')
                         .str.replace('stimulus_exp_m/', 'Mean_')
                         .str.replace('stimulus_exp_v/', 'Vocal_'))

        # Group emotion labels (replace word with valence tag)
        posemo = ['PRETTY', 'SATISFACT', 'REFRESH', 'NATURAL', 'WARM', 'CALMNESS', 'GRATEFUL', 'NEW']
        negemo = ['FATIGUE', 'SLY', 'ANXIETY', 'COMPLAINT', 'DISLIKE', 'SORE', 'TASTELESS', 'BITTER']
        for e in posemo:
            label['type'] = label['type'].str.replace(e, 'P-')
        for e in negemo:
            label['type'] = label['type'].str.replace(e, 'N-')

        # === Annotate events into raw ===
        onset = label['latency']  # Already in seconds
        duration = np.repeat(0.0, len(label))  # Instantaneous events
        description = label['type']
        raw.set_annotations(mne.Annotations(onset=onset, duration=duration, description=description))

        # === Define event ID mapping ===
        event_id = {
            'Mean_P-P': 1, 'Mean_P-N': 2, 'Mean_N-P': 3, 'Mean_N-N': 4,
            'Vocal_P-P': 5, 'Vocal_P-N': 6, 'Vocal_N-P': 7, 'Vocal_N-N': 8
        }

        # === Extract MNE events ===
        events, _ = mne.events_from_annotations(raw, event_id=event_id)

        # === Event sanity check ===
        if len(events) <= 64:
            print(f"✅ Extracting epochs for subject {subjid}...")

            # Unpack timing and baseline
            tmin = epoch_params[mode]['tmin']
            tmax = epoch_params[mode]['tmax']
            baseline = epoch_params[mode]['baseline']

            # Construct Epochs
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                # reject=reject_criteria if mode == 'erp' else None,
                reject_by_annotation=True,
                event_repeated='merge',
                on_missing='ignore',
                preload=True
            )

            print(epochs)

            # Save output epochs
            save_path = os.path.join('data', f'Att_VS_{mode}_{subjid}_epo.fif.gz')
            epochs.save(save_path, overwrite=True)

        else:
            print(f"⚠️ Too many events for subject {subjid} (n={len(events)}); skipping...")
