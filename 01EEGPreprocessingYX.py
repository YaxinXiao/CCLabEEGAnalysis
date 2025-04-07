"""
EEG Data Preprocessing Pipeline

This script performs preprocessing on EEG data including:
- Average reference application
- Bad channel detection and interpolation
- Band-pass filtering
- ICA decomposition and automatic component rejection
- Muscle artifact detection and annotation
- Saving cleaned EEG data

Dependencies: mne, numpy, pandas, matplotlib, mne_icalabel

Preprocessing Summary:
- Reference: Applied before filtering and ICA.
  Average reference is applied to ensure consistent spatial structure for ICA and filtering.
- Filtering: Applied.
  Band-pass filtering (1–63.9 Hz) is directly applied to the raw data.
- Bad channels: Applied.
  Bad channels are detected using a z-score threshold and then interpolated.
- ICA: Applied.
  ICA is fitted to the cleaned data and non-brain components are removed using ICLabel.
  The cleaned signal is stored in 'reconst_raw'.
- Bad segments: Annotated only.
  Muscle artifacts are detected and annotated, but not yet excluded.
  These annotations will be used during epoching (e.g., with reject_by_annotation=True).

Author: Yaxin Xiao
Date: March 2025
"""

import os
from os.path import split
from glob import glob
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA, annotate_muscle_zscore
from mne_icalabel import label_components

# =========================
# ==== USER PARAMETERS ====
# =========================
# Data paths -- should have been created already
input_folder = 'vs0318'
output_data_folder = 'data' # for output data
output_fig_folder = 'figure' # for ICA figures

# Channel list used in your cap
ch_names = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
            'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

# Filtering (band-pass)
l_freq = 1.0
h_freq = 63.9

# Bad channel detection threshold (z-score of channel std)
bad_channel_z_thresh = 3.0

# ICA parameters
ica_n_components = 0.99       # Keep 99% of variance
ica_method = 'infomax'
ica_random_state = 97

# Muscle artifact detection threshold
muscle_z_thresh = 4.0

# =========================
# ===== MAIN SCRIPT =======
# =========================

# Create folders if they don't exist
os.makedirs(output_data_folder, exist_ok=True)
os.makedirs(output_fig_folder, exist_ok=True)

# Get all EEG .edf files in the input folder
eeg_files = sorted(glob(os.path.join(input_folder, '*VS*edf')))

# Function to detect bad channels using z-score of channel standard deviations
def detect_bad_channels_zscore(raw, threshold=3.0):
    raw_tmp = raw.copy().pick_types(eeg=True)
    data = raw_tmp.get_data()
    stds = np.std(data, axis=1)
    zscores = (stds - np.mean(stds)) / np.std(stds)
    bad_idx = np.where(np.abs(zscores) > threshold)[0]
    bads = [raw_tmp.ch_names[i] for i in bad_idx]
    return bads

# Process each EEG file
for i, ef in enumerate(eeg_files):
    print(f"\n---------------\nProcessing file: {ef}\n---------------")

    # Extract subject ID
    subjid = split(ef)[1].split('_')[4]

    # Load EEG file
    raw = mne.io.read_raw_edf(ef, preload=True)

    # Drop channels not in specified cap
    ch_df = pd.Series(raw.ch_names)
    ch_drop = list(ch_df[~ch_df.isin(ch_names)])
    raw.drop_channels(ch_drop)

    # Set standard montage
    raw.set_montage('easycap-M1')

    # ===== APPLY AVERAGE REFERENCE BEFORE FILTERING =====
    raw.set_eeg_reference("average")

    # Band-pass filter
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Detect bad channels
    bads = detect_bad_channels_zscore(raw, threshold=bad_channel_z_thresh)
    raw.info['bads'] = bads
    print(f"Detected bad channels: {bads}")
    raw.interpolate_bads()

    # ICA decomposition
    raw.set_eeg_reference("average")
    ica = ICA(
        n_components=ica_n_components,
        max_iter="auto",
        method=ica_method,
        random_state=ica_random_state,
        fit_params=dict(extended=True)
    )
    ica.fit(raw)

    # Plot ICA components and save figures
    component_figs = ica.plot_components(show=False)
    if isinstance(component_figs, list):
        for i, fig in enumerate(component_figs):
            fig.savefig(os.path.join(output_fig_folder, f'ICA_components_{subjid}_page{i + 1}.png'))
    else:
        component_figs.savefig(os.path.join(output_fig_folder, f'ICA_components_{subjid}_page1.png'))

    # Auto-label ICA components
    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"Excluding ICA components: {exclude_idx} | Labels: {[labels[e] for e in exclude_idx]}")

    # Apply ICA exclusion to raw data
    reconst_raw = ica.apply(raw.copy(), exclude=exclude_idx)

    # Annotate muscle artifacts
    reconst_raw_resampled = reconst_raw.copy().resample(512)
    annotations, scores = annotate_muscle_zscore(reconst_raw_resampled, ch_type="eeg", threshold=muscle_z_thresh)
    reconst_raw.set_annotations(annotations)
    print(f"Annotated {len(annotations)} bad segments.")

    # Save preprocessed data
    reconst_raw.save(os.path.join(output_data_folder, f'Att_VS_ICA_{subjid}.fif.gz'), overwrite=True)
