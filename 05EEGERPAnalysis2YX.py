"""
ERPanalysis.py

This script loads the compiled ERP dataframe (output of ERP-compute.py), performs:
- Line plots of ERP by condition and group (all selected channels)
- Difference wave and significance marking (ttest per timepoint)
- Mean_diff vs Vocal_diff plots across all channels with significance marking

Author: Yaxin Xiao
Date: April 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# =========================
# ==== USER PARAMETERS ====
# =========================

input_csv = 'data/df_erp_all.csv'
subj_info_csv = 'ParticipantSelfCon.csv'
channels = ['Fz', 'Cz', 'Pz', 'Oz', 'Fp1', 'Fp2', 'F7', 'F3', 'FC1', 'C3', 'FC5',
            'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'O2', 'PO10',
            'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8']
diff_pairs = [('Mean_Incong', 'Mean_Cong'), ('Vocal_Incong', 'Vocal_Cong')]
groups = {'euro': 'European American', 'asia': 'East Asian'}

# # Output folders
# os.makedirs('results', exist_ok=True)
# os.makedirs('results/condition_by_channel', exist_ok=True)
os.makedirs('results/ERPallchannels', exist_ok=True)

# =========================
# ==== LOAD DATA ====
# =========================

dfall = pd.read_csv(input_csv)
subjinf = pd.read_csv(subj_info_csv)
times = np.sort(dfall['time'].unique())

# =========================
# ==== PLOTTING ====
# =========================

for code, label in groups.items():
    gid = subjinf[subjinf['Ethnicity Group'] == label]['ID'].values
    groupdf = dfall[dfall['subjid'].isin(gid)]

    selected_channels = [ch for ch in channels if ch in groupdf['loc'].unique()]
    ncols = 6
    nrows = int(np.ceil(len(selected_channels) / ncols))

    # === Plot 1: All 4 Conditions ===
    fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes1 = axes1.flatten()

    for i, ch in enumerate(selected_channels):
        chdf = groupdf[groupdf['loc'] == ch]
        sns.lineplot(data=chdf, x='time', y='uV', hue='cond', ax=axes1[i], estimator='mean', errorbar='se',
                     hue_order=['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong'])
        axes1[i].set_title(ch)
        axes1[i].axvline(0, color='black', linestyle='--')
        axes1[i].axhline(0, color='black', linestyle='--')

    handles1, labels1 = axes1[0].get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc='upper center', ncol=4)
    fig1.suptitle(f'{label} ERP - 4 Conditions', fontsize=18)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(f'results/ERPallchannels/ERP_4cond_{code}.png')
    plt.close()

    # === Plot 2: Difference (Mean - Cong) with significance ===
    fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes2 = axes2.flatten()

    for i, ch in enumerate(selected_channels):
        chdf = groupdf[groupdf['loc'] == ch]
        pivot = chdf.pivot_table(index=['time', 'subjid'], columns='cond', values='uV').dropna()
        if pivot.empty or any(c not in pivot.columns for c in ['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong']):
            continue

        ax = axes2[i]
        for k, (incong, cong) in enumerate(diff_pairs):
            diff = pivot[incong] - pivot[cong]
            diff_avg = diff.groupby('time').mean()
            diff_sem = diff.groupby('time').sem()

            ax.plot(times, diff_avg, label=f'{incong} - {cong}', linestyle='-', alpha=0.9)
            ax.fill_between(times, diff_avg - diff_sem, diff_avg + diff_sem, alpha=0.2)

            pvals = [ttest_rel(pivot.loc[t][incong], pivot.loc[t][cong])[1] if t in pivot.index else np.nan for t in times]
            pvals = np.array(pvals)
            sig_mask = pvals < 0.05
            for idx in np.where(sig_mask)[0]:
                ax.axvspan(times[idx], times[idx], color='green' if k == 0 else 'red', alpha=0.15)

        ax.set_title(ch)
        ax.axvline(0, color='black', linestyle='--')
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlim(times[0], times[-1])

    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper center', ncol=2)
    fig2.suptitle(f'{label} ERP Diff (w/ significance)', fontsize=18)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(f'results/ERPallchannels/ERP_diffsig_{code}.png')
    plt.close()

    print(f"✅ Saved 4-condition + diff-significance ERP plots for {code}")
