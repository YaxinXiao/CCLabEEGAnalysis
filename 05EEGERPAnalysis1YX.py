"""
ERPanalysis.py

This script loads the compiled ERP dataframe (output of ERP-compute.py), performs:
- Line plots of ERP by condition and group (Fz, Cz, Pz)
- Difference wave and significance marking (ttest per timepoint)
- Time window analysis (paired t-tests)
- Cluster-based permutation test for Mean_diff - Vocal_diff

Author: Yaxin Xiao
Date: April 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from mne.stats import permutation_cluster_1samp_test

# =========================
# ==== USER PARAMETERS ====
# =========================

input_csv = 'data/df_erp_all.csv'
subj_info_csv = 'ParticipantSelfCon.csv'
channels = [ 'Fz','Cz','Pz', 'Oz']
# channels_of_interest = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
#             'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6',
#             'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
diff_pairs = [('Mean_Incong', 'Mean_Cong'), ('Vocal_Incong', 'Vocal_Cong')]
groups = {'euro': 'European American', 'asia': 'East Asian'}

# Output folders
# os.makedirs('results', exist_ok=True)
os.makedirs('results/ERPSelectedChannels', exist_ok=True)

# =========================
# ==== LOAD DATA ====
# =========================

dfall = pd.read_csv(input_csv)
subjinf = pd.read_csv(subj_info_csv)
times = np.sort(dfall['time'].unique())

# =========================
# ==== ERP PLOTTING ====
# =========================
for code, label in groups.items():
    gid = subjinf[subjinf['Ethnicity Group'] == label]['ID'].values
    groupdf = dfall[dfall['subjid'].isin(gid)]

    n_ch = len(channels)
    fig, axes = plt.subplots(2, n_ch + 1, figsize=(5 * (n_ch + 1), 8), sharex='col', sharey='row',
                             gridspec_kw={'width_ratios': [1] * n_ch + [0.15]})
    fig.suptitle(f'{label} ERP - Conditions & Differences', fontsize=20, y=0.97)

    for i, ch in enumerate(channels):
        chdf = groupdf[groupdf['loc'] == ch]

        # ==== First Row: 4-condition plot ====
        sns.lineplot(data=chdf, x='time', y='uV', hue='cond', ax=axes[0, i], errorbar='se', estimator='mean',
                     hue_order=['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong'])
        axes[0, i].set_title(f'{ch} - Conditions')
        axes[0, i].axvline(0, color='k', linestyle='--')
        axes[0, i].axhline(0, color='k', linestyle='--')
        axes[0, i].set_xlim(-0.5, 1.5)

        # ==== Second Row: diff and significance ====
        pivot = chdf.pivot_table(index=['time', 'subjid'], columns='cond', values='uV').dropna()
        if pivot.empty:
            continue

        for j, (incong, cong) in enumerate(diff_pairs):
            diff = pivot[incong] - pivot[cong]
            mean_diff = diff.groupby('time').mean()
            sem_diff = diff.groupby('time').sem()

            color = 'blue' if j == 0 else 'orange'
            label_diff = f'{incong} - {cong}'
            axes[1, i].plot(times, mean_diff, label=label_diff, color=color)
            axes[1, i].fill_between(times, mean_diff - sem_diff, mean_diff + sem_diff, alpha=0.3, color=color)

            pvals = [ttest_rel(pivot.loc[t][incong], pivot.loc[t][cong])[1] if t in pivot.index else np.nan for t in times]
            sig_mask = np.array(pvals) < 0.05
            for idx in np.where(sig_mask)[0]:
                axes[1, i].axvspan(times[idx], times[idx], color=color, alpha=0.15)

        axes[1, i].set_title(f'{ch} - Diff & Sig')
        axes[1, i].axvline(0, color='k', linestyle='--')
        axes[1, i].axhline(0, color='k', linestyle='--')
        axes[1, i].set_xlim(-0.5, 1.5)

    # Legend on right side of each row
    for row_idx, legend_labels in enumerate([
        ['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong'],
        ['Mean_Incong - Mean_Cong', 'Vocal_Incong - Vocal_Cong']
    ]):
        handles, labels_ = [], []
        if row_idx == 0:
            palette = sns.color_palette(n_colors=4)
            for c, lbl in zip(palette, legend_labels):
                handles.append(plt.Line2D([], [], color=c, label=lbl))
        else:
            handles = [
                plt.Line2D([], [], color='blue', label='Mean_Incong - Mean_Cong'),
                plt.Line2D([], [], color='orange', label='Vocal_Incong - Vocal_Cong')
            ]

        axes[row_idx, -1].legend(handles=handles, loc='center', frameon=False)
        axes[row_idx, -1].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'results/ERPSelectedChannels/ERP_SelectedChan_{code}.png')
    plt.close()
    print(f"✅ Saved combined ERP grid plot for {code}")

# =========================
# ==== TIME WINDOW STATS ====
# =========================

time_windows = [(0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.75),
                (0.75, 0.9), (0.9, 1.05), (1.05, 1.2), (1.2, 1.35), (1.35, 1.5)]

for code, label in groups.items():
    gid = subjinf[subjinf['Ethnicity Group'] == label]['ID'].values
    groupdf = dfall[dfall['subjid'].isin(gid)]
    results = []

    for ch in channels:
        chdf = groupdf[groupdf['loc'] == ch]
        for (t_start, t_end) in time_windows:
            mask_time = (chdf['time'] >= t_start) & (chdf['time'] < t_end)
            tw_data = chdf[mask_time]
            if tw_data.empty:
                continue
            grp = tw_data.groupby(['subjid', 'cond'])['uV'].mean().reset_index()
            pivot_df = grp.pivot(index='subjid', columns='cond', values='uV')
            pivot_df = pivot_df.dropna(subset=['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong'], how='any')
            pivot_df['Mean_diff'] = pivot_df['Mean_Incong'] - pivot_df['Mean_Cong']
            pivot_df['Vocal_diff'] = pivot_df['Vocal_Incong'] - pivot_df['Vocal_Cong']
            if len(pivot_df) > 1:
                tstat, pval = ttest_rel(pivot_df['Mean_diff'], pivot_df['Vocal_diff'])
            else:
                tstat, pval = np.nan, np.nan
            results.append({
                'channel': ch,
                'time_window': f'{int(t_start * 1000)}-{int(t_end * 1000)} ms',
                'n_subjects': len(pivot_df),
                'Mean_diff_avg': pivot_df['Mean_diff'].mean(),
                'Vocal_diff_avg': pivot_df['Vocal_diff'].mean(),
                't_stat': tstat,
                'p_value': pval
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['channel', 'time_window']).reset_index(drop=True)
    outfile = f'results/ERPSelectedChannels/results_{code}.csv'
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(outfile, index=False)
    print(f"📊 Time-window results saved: {outfile}")

# =========================
# ==== CLUSTER ANALYSIS ====
# =========================

for code, label in groups.items():
    gid = subjinf[subjinf['Ethnicity Group'] == label]['ID'].values
    groupdf = dfall[dfall['subjid'].isin(gid)]

    for ch in channels:
        chdf = groupdf[groupdf['loc'] == ch]
        subj_ids = chdf['subjid'].unique()
        subj_ids.sort()
        all_diffwaves = []

        for sid in subj_ids:
            sdf = chdf[chdf['subjid'] == sid]
            pivot = sdf.pivot_table(index='time', columns='cond', values='uV')
            try:
                mean_diff = pivot['Mean_Incong'] - pivot['Mean_Cong']
                vocal_diff = pivot['Vocal_Incong'] - pivot['Vocal_Cong']
                diff = mean_diff - vocal_diff
                all_diffwaves.append(diff.values)
            except KeyError:
                continue

        if len(all_diffwaves) < 2:
            continue

        data = np.array(all_diffwaves)
        T_obs, clusters, pvals, H0 = permutation_cluster_1samp_test(
            data, threshold=None, tail=0, n_permutations=1000,
            out_type='mask', verbose=False
        )

        plt.figure(figsize=(12, 5))
        plt.plot(times, data.mean(axis=0), label='(Mean_diff - Vocal_diff)', color='orange')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('uV')
        plt.title(f'{label} | {ch} | Cluster test (p<0.05 shaded)')

        for i_c, c in enumerate(clusters):
            if pvals[i_c] < 0.05:
                plt.axvspan(times[c][0], times[c][-1], color='red', alpha=0.3)

        plt.legend()
        outfig = f'results/ERPSelectedChannels/{code}_{ch}_cluster.png'
        plt.savefig(outfig)
        plt.close()
        print(f"✅ Cluster figure saved: {outfig}")

