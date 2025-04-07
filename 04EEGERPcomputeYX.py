"""
ERP-compute.py

This script loads subject-wise ERP epoch data (.fif), extracts mean uV by condition,
removes outliers, and compiles a long-format DataFrame for downstream analysis.
The output is saved as a CSV for use in ERPanalysis.py.

Author: Yaxin Xiao
Date: April 2025
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

# ERP conditions of interest
cond = [
    'Mean_P-P', 'Mean_P-N', 'Mean_N-P', 'Mean_N-N',
    'Vocal_P-P', 'Vocal_P-N', 'Vocal_N-P', 'Vocal_N-N'
]

# Channels to include in final analysis (optional filtering)
# channels_of_interest = ['Fz', 'Cz', 'Pz']
channels_of_interest = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
            'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
            'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

# Input/output paths
data_path = 'data/Att_VS_erp_*.fif.gz'
output_csv = 'data/df_erp_all.csv'

# =========================
# ==== LOAD AND EXTRACT ====
# =========================

files = sorted(glob(data_path))
dfall = []

for f in files:
    epochs = mne.read_epochs(f, preload=True)
    subjid = int(re.findall(r'\d+', os.path.basename(f))[0])
    evokedf = pd.DataFrame()

    for c in cond:
        if c in epochs.event_id and len(epochs[c]) > 0:
            df = epochs[c].to_data_frame()
            df['condition'] = c
            evokedf = pd.concat([evokedf, df], ignore_index=True)
        else:
            print(f"⚠️ Skipping subject {subjid}, condition {c}: empty or missing")

    if evokedf.empty:
        print(f"❌ Skipping subject {subjid}: no valid conditions found")
        continue

    # Ensure correct epochs only
    evokedf = evokedf[evokedf['epoch'].isin(epochs.selection)]

    # Long format: each row = timepoint x condition x channel x subject
    avedat = evokedf.melt(id_vars=['time', 'condition', 'epoch'],
                          var_name='loc', value_name='uV')

    # Remove outliers per channel
    avedat_out = []
    for ch in avedat['loc'].unique():
        subdf = avedat[avedat['loc'] == ch]
        m, s = subdf['uV'].mean(), subdf['uV'].std()
        avedat_out.append(subdf[(subdf['uV'] > m - 3 * s) & (subdf['uV'] < m + 3 * s)])
    avedat = pd.concat(avedat_out)

    avedat['subjid'] = subjid
    dfall.append(avedat)

# =========================
# ==== COMBINE & CLEAN ====
# =========================

print("✅ Finished extracting data from all subjects")
dfall = pd.concat(dfall, ignore_index=True)
dfall.rename(columns={'condition': 'cond'}, inplace=True)

# Relabel condition to Cong vs Incong
# Cong: P-P, N-N | Incong: P-N, N-P
dfall['cond'] = (dfall['cond'].str.replace('P-P', 'Cong')
                               .str.replace('N-N', 'Cong')
                               .str.replace('P-N', 'Incong')
                               .str.replace('N-P', 'Incong'))

# Optional: only retain selected channels
if channels_of_interest:
    dfall = dfall[dfall['loc'].isin(channels_of_interest)]

# =========================
# ==== SAVE OUTPUT CSV ====
# =========================

dfall.to_csv(output_csv, index=False)
print(f"📁 Saved combined ERP data to {output_csv}")
