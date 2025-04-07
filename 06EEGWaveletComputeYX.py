# ===============================
# EEG Wavelet Analysis Pipeline: compute_wavelet
# Author: Yaxin Xiao
# Description: Computes time-frequency representations (TFR) using Morlet wavelets
# ===============================

import copy
import numpy as np
import pandas as pd
import mne
from glob import glob
from mne.time_frequency import tfr_array_morlet, EpochsTFR

# =========================
# ==== USER PARAMETERS ====
# =========================
sr = 128  # sampling rate
base = 'logratio'  # baseline correction mode: 'logratio', 'mean', 'ratio', etc.
basetime = (-0.2, -0.05)  # baseline time window in seconds
decim_time = 10  # decimation factor to reduce time resolution
cycle = 5  # number of wavelet cycles
freq_mode = 'linear'  # options: 'linear', 'log', or 'classic'

# Frequency setup
if freq_mode == 'linear':
    freqs = np.arange(4, 48.5, 0.5)
elif freq_mode == 'log':
    freqs = np.logspace(np.log10(4), np.log10(50), 30)
elif freq_mode == 'classic':
    freqs = np.array([4, 5.5, 7, 8.5, 10, 12, 15, 18, 21, 25, 30, 35, 40, 45, 50])
else:
    raise ValueError("Invalid freq_mode. Choose from 'linear', 'log', or 'classic'.")

# Define frequency bands based on freqs
def get_band_idx(freqs, fmin, fmax):
    return np.where((freqs >= fmin) & (freqs < fmax))[0]

band_idx = {
    'delta': get_band_idx(freqs, 1, 4),
    'theta': get_band_idx(freqs, 4, 8),
    'lower_alpha': get_band_idx(freqs, 8, 10),
    'upper_alpha': get_band_idx(freqs, 10, 13),
    'beta': get_band_idx(freqs, 13, 30),
    'gamma': get_band_idx(freqs, 30, 48.5)
}

# Input files
files = sorted(glob('data/Att_VS_wavelet_*.fif.gz'))

# =========================
# ==== WAVELET LOOP =======
# =========================

for f in files:
    print(f"\n📂 Processing file: {f}")
    mnedat = mne.read_epochs(f, preload=True)
    df = mnedat.to_data_frame()
    info = mnedat.info
    ch_names = info.ch_names
    times = df['time'].unique()

    condtfr = []
    tfrcomment = []

    # Loop through conditions
    for ii, c in enumerate(df['condition'].unique()):
        print(f"  ➤ Condition: {c}")
        conddf = df.query('condition==@c')
        epoinf = conddf['epoch'].unique()
        elen = len(epoinf)

        # Create epochs matrix: shape (n_epochs, n_channels, n_times)
        epochs = np.zeros([elen, len(ch_names), len(times)])
        for i, e in enumerate(epoinf):
            epochs[i, :, :] = conddf.query('epoch==@e')[ch_names].T

        # Compute wavelet transform (power)
        tfr = tfr_array_morlet(epochs, sfreq=sr, freqs=freqs, n_cycles=cycle,
                               output='power', n_jobs=-1)

        # Accumulate across conditions
        if ii == 0:
            condtfr = copy.deepcopy(tfr)
            tfrcomment = np.repeat(c, elen)
        else:
            condtfr = np.concatenate([condtfr, tfr])
            tfrcomment = np.concatenate([tfrcomment, np.repeat(c, elen)])

    # Convert to MNE TFR object
    mnetfr = EpochsTFR(info, condtfr, times, freqs, comment='wavelet', method='morlet')

    # Apply baseline correction and decimation
    mnetfr.apply_baseline(mode=base, baseline=basetime)
    mnetfr.decimate(decim_time)

    # Save TFR object and condition label file
    out_tfr = f.replace('epo.fif.gz', 'epo-tfr.h5')
    out_csv = f.replace('epo.fif.gz', 'epo-tfr.csv')
    mnetfr.save(out_tfr, overwrite=True)
    pd.DataFrame(tfrcomment, columns=['condition']).to_csv(out_csv, index=False)
    print(f"  ✅ Saved: {out_tfr} & {out_csv}")
