# ===============================
# EEG Wavelet Time-Frequency Plotting & Cluster Testing
# Author: Yaxin Xiao
# Description: Loads extracted TFRs, calculates condition-wise averages,
# performs cluster-based permutation tests, and visualizes results
# ===============================

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from mne.time_frequency import read_tfrs
from mne.stats import permutation_cluster_test
from scipy.stats import ttest_rel

# =========================
# ==== USER PARAMETERS ====
# =========================
def get_band_idx(freqs, fmin, fmax):
    return np.where((freqs >= fmin) & (freqs < fmax))[0]

subjinf = pd.read_csv('ParticipantSelfCon.csv')
target_channels = ['Fz', 'Cz', 'Pz']
data_dir = 'data'
out_dir = 'results/wavelettest'
os.makedirs(out_dir, exist_ok=True)

# =========================
# ==== LOAD FREQUENCIES & BAND DEFINITIONS ====
# =========================
first_tfr_file = sorted(glob(os.path.join(data_dir, 'Att_VS_*_epo-tfr.h5')))[0]
spectemp0 = read_tfrs(first_tfr_file)
if isinstance(spectemp0, list):
    spectemp0 = spectemp0[0]
freqs = spectemp0.freqs  # actual frequency array from TFR file

band_idx = {
    'delta': get_band_idx(freqs, 1, 4),
    'theta': get_band_idx(freqs, 4, 8),
    'lower_alpha': get_band_idx(freqs, 8, 10),
    'upper_alpha': get_band_idx(freqs, 10, 13),
    'alpha': get_band_idx(freqs, 8, 13),
    'beta': get_band_idx(freqs, 13, 30),
    'gamma': get_band_idx(freqs, 30, 50.5)
}

# =========================
# ==== LOAD AND EXTRACT TFR ====
# =========================
condtf = pd.DataFrame()
files = sorted(glob(os.path.join(data_dir, 'Att_VS_*_epo-tfr.h5')))

for f in files:
    subjid = int(re.findall(r'\d+', f)[0])
    spectemp = read_tfrs(f)
    if isinstance(spectemp, list):
        spectemp = spectemp[0]

    tfr = spectemp.data  # shape: (n_epochs, n_channels, n_freqs, n_times)
    times = pd.Series(spectemp.times)
    cond = pd.read_csv(f.replace('.h5', '.csv'))
    cond['cond'] = cond['condition'].str.replace('P-P', 'Cong').str.replace('N-N', 'Cong') \
                                     .str.replace('P-N', 'Incong').str.replace('N-P', 'Incong')

    for i, ch in enumerate(spectemp.ch_names):
        for co in cond['cond'].unique():
            condindex = cond.query('cond==@co').index
            if condindex.empty:
                continue
            tf = np.mean(tfr[condindex, i, :, :], axis=0)  # avg across epochs
            t = pd.DataFrame(tf.T, columns=[f'band{n}' for n in range(tf.shape[0])])
            condtf = pd.concat([condtf, t.assign(cond=co, ch=ch, time=times, subjid=subjid)], ignore_index=True)

condtf.to_pickle(os.path.join(data_dir, 'condtf.pkl.gz'))
print("✅ Saved TFR summary to condtf.pkl.gz")


# ----------------------- Time-Frequency Plots for Fz, Cz, Pz (Diff Conditions) ----------------------- #
subjinf = pd.read_csv('ParticipantSelfCon.csv')
target_channels = ['Fz', 'Cz', 'Pz']
time_vals = sorted(condtf['time'].unique())
freq_vals = np.arange(47)

all_diffs = []  # Collect for global color scaling

for row_idx, (group, prefixes) in enumerate([('euro', ['Mean']), ('euro', ['Vocal']), ('asia', ['Mean']), ('asia', ['Vocal'])]):
    ethnicity = 'European American' if group == 'euro' else 'East Asian'
    gid = subjinf[subjinf['Ethnicity Group'] == ethnicity]['ID'].values
    prefix = prefixes[0]
    cong_label = f"{prefix}_Cong"
    incong_label = f"{prefix}_Incong"

    for col_idx, ch in enumerate(target_channels):
        df_cong = condtf[(condtf['subjid'].isin(gid)) & (condtf['ch'] == ch) & (condtf['cond'] == cong_label)]
        df_incong = condtf[(condtf['subjid'].isin(gid)) & (condtf['ch'] == ch) & (condtf['cond'] == incong_label)]

        df_cong = df_cong.sort_values('time')
        df_incong = df_incong.sort_values('time')

        cong_power = df_cong[[f'band{i}' for i in range(47)]].groupby(df_cong['time']).mean().T
        incong_power = df_incong[[f'band{i}' for i in range(47)]].groupby(df_incong['time']).mean().T

        diff_power = incong_power - cong_power
        all_diffs.append(diff_power.values)

# Determine global color scale based on all diffs
all_diffs_array = np.stack(all_diffs)
vmin = np.nanpercentile(all_diffs_array, 1)
vmax = np.nanpercentile(all_diffs_array, 99)

# Plot again with colorbar placed on the right of the last column
fig, axs = plt.subplots(4, 3, figsize=(18, 16), sharex=True, sharey=True)
fig.suptitle('Time-Frequency Power (Condition Differences)', fontsize=24)

plot_idx = 0
ims = []
for row_idx, (group, prefixes) in enumerate([('euro', ['Mean']), ('euro', ['Vocal']), ('asia', ['Mean']), ('asia', ['Vocal'])]):
    ethnicity = 'European American' if group == 'euro' else 'East Asian'
    gid = subjinf[subjinf['Ethnicity Group'] == ethnicity]['ID'].values
    prefix = prefixes[0]
    cong_label = f"{prefix}_Cong"
    incong_label = f"{prefix}_Incong"

    for col_idx, ch in enumerate(target_channels):
        df_cong = condtf[(condtf['subjid'].isin(gid)) & (condtf['ch'] == ch) & (condtf['cond'] == cong_label)]
        df_incong = condtf[(condtf['subjid'].isin(gid)) & (condtf['ch'] == ch) & (condtf['cond'] == incong_label)]

        df_cong = df_cong.sort_values('time')
        df_incong = df_incong.sort_values('time')

        cong_power = df_cong[[f'band{i}' for i in range(47)]].groupby(df_cong['time']).mean().T
        incong_power = df_incong[[f'band{i}' for i in range(47)]].groupby(df_incong['time']).mean().T

        diff_power = incong_power - cong_power

        im = axs[row_idx, col_idx].imshow(diff_power.values, aspect='auto', origin='lower',
                                          extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]],
                                          cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[row_idx, col_idx].set_title(f"{ethnicity} - {prefix} - {ch}")
        axs[row_idx, col_idx].axvline(0, color='black', linestyle='--')
        axs[row_idx, col_idx].set_xlabel('Time (s)')
        axs[row_idx, col_idx].set_ylabel('Freq Index')
        ims.append(im)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(ims[0], cax=cbar_ax)
cbar.set_label('Power Diff (dB)')

fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])
plt.savefig(f'{out_dir}/TimeFreq_Group_Diff_FzCzPz.png')
plt.close()

# Statistical Comparison: Mean_Diff vs Vocal_Diff
tval_maps = {}
pval_maps = {}

for group in ['euro', 'asia']:
    ethnicity = 'European American' if group == 'euro' else 'East Asian'
    gid = subjinf[subjinf['Ethnicity Group'] == ethnicity]['ID'].values

    for ch in target_channels:
        mean_diffs, vocal_diffs = [], []

        for sid in gid:
            def get_diff_df(cond1, cond2):
                df1 = condtf.query("subjid == @sid and ch == @ch and cond == @cond1").sort_values('time')
                df2 = condtf.query("subjid == @sid and ch == @ch and cond == @cond2").sort_values('time')
                df1 = df1.set_index('time').reindex(time_vals)
                df2 = df2.set_index('time').reindex(time_vals)
                diff = df2[[f'band{i}' for i in range(47)]].values - df1[[f'band{i}' for i in range(47)]].values
                return diff.T

            m_diff = get_diff_df('Mean_Cong', 'Mean_Incong')
            v_diff = get_diff_df('Vocal_Cong', 'Vocal_Incong')

            if m_diff.shape == (47, len(time_vals)) and v_diff.shape == (47, len(time_vals)):
                mean_diffs.append(m_diff)
                vocal_diffs.append(v_diff)

        mean_arr = np.stack(mean_diffs)  # subj × freq × time
        vocal_arr = np.stack(vocal_diffs)

        tvals = np.zeros((47, len(time_vals)))
        pvals = np.zeros((47, len(time_vals)))

        for f in range(47):
            for t in range(len(time_vals)):
                tval, pval = ttest_rel(vocal_arr[:, f, t], mean_arr[:, f, t], nan_policy='omit')
                tvals[f, t] = tval
                pvals[f, t] = pval

        tval_maps[f'{group}_{ch}'] = tvals
        pval_maps[f'{group}_{ch}'] = pvals

# Plot Significant Regions (p < 0.05)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Significant Vocal vs Mean Diff Regions (p < 0.05)", fontsize=18)

for i, group in enumerate(['euro', 'asia']):
    for j, ch in enumerate(target_channels):
        sig_mask = (pval_maps[f'{group}_{ch}'] < 0.05).astype(int)

        axs[i, j].imshow(sig_mask, aspect='auto', origin='lower',
                         extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]], cmap='Reds')
        axs[i, j].set_title(f"{'Euro' if group == 'euro' else 'Asian'} - {ch}")
        axs[i, j].axvline(0, color='black', linestyle='--')
        axs[i, j].set_xlabel("Time (s)")
        axs[i, j].set_ylabel("Freq Index")

fig.tight_layout()
plt.savefig(f"{out_dir}/Significant_Vocal_vs_Mean.png")
plt.close()

# =========================
# ==== CLUSTER TEST (Mean vs Vocal Diff) ====
# =========================
# time_vals = sorted(condtf['time'].unique())
# freq_vals = np.arange(47)
#
# for group in ['euro', 'asia']:
#     ethnicity = 'European American' if group == 'euro' else 'East Asian'
#     gid = subjinf[subjinf['Ethnicity Group'] == ethnicity]['ID'].values
#
#     for ch in target_channels:
#         print(f"🔍 {ethnicity} - {ch}")
#         mean_diffs, vocal_diffs = [], []
#
#         for sid in gid:
#             def get_diff_df(cond1, cond2):
#                 df1 = condtf.query("subjid == @sid and ch == @ch and cond == @cond1").sort_values('time')
#                 df2 = condtf.query("subjid == @sid and ch == @ch and cond == @cond2").sort_values('time')
#                 df1 = df1.set_index('time').reindex(time_vals)
#                 df2 = df2.set_index('time').reindex(time_vals)
#                 return (df2[[f'band{i}' for i in range(47)]].values - df1[[f'band{i}' for i in range(47)]].values).T
#
#             m_diff = get_diff_df('Mean_Cong', 'Mean_Incong')
#             v_diff = get_diff_df('Vocal_Cong', 'Vocal_Incong')
#
#             if m_diff.shape == (47, len(time_vals)) and v_diff.shape == (47, len(time_vals)):
#                 mean_diffs.append(m_diff)
#                 vocal_diffs.append(v_diff)
#
#         if not mean_diffs or not vocal_diffs:
#             print("⚠️ Not enough data for cluster test")
#             continue
#
#         mean_arr = np.stack(mean_diffs)
#         vocal_arr = np.stack(vocal_diffs)
#
#         T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
#             [mean_arr, vocal_arr], n_permutations=1000, tail=0, out_type='mask', n_jobs=1)
#
#         sig_mask = np.zeros_like(T_obs, dtype=float)
#         for cl, p in zip(clusters, cluster_p_values):
#             sig_mask[cl] = 2 if p < 0.05 else (1 if p < 0.1 else 0)
#
#         cmap = plt.cm.colors.ListedColormap(['white', '#fca4b6', '#e41a1c'])
#         plt.figure(figsize=(10, 5))
#         plt.imshow(sig_mask, aspect='auto', origin='lower',
#                    extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]],
#                    cmap=cmap, vmin=0, vmax=2)
#         plt.title(f"{ethnicity} - {ch} (Cluster p<0.1)")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Freq Index")
#         plt.axvline(0, color='black', linestyle='--')
#         plt.tight_layout()
#         plt.savefig(f'{out_dir}/ClusterSig_{group}_{ch}.png')
#         plt.close()
#         print(f"✅ Saved: ClusterSig_{group}_{ch}.png")


# ----------------------- Custom Alpha Power Plot for Fz, Cz, Pz ----------------------- #
subjinf = pd.read_csv('ParticipantSelfCon.csv')
idvar = ['cond', 'ch', 'time']
horder = ['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong']
target_channels = ['Fz', 'Cz', 'Pz']
band = 'upper_alpha'
start, end = band_idx[band]
alpha_cols = [f'band{n}' for n in range(start, end)]

for group in ['euro', 'asia']:
    ethnicity = 'European American' if group == 'euro' else 'East Asian'
    gid = subjinf[subjinf['Ethnicity Group'] == ethnicity]['ID'].values
    group_data = condtf[condtf['subjid'].isin(gid) & condtf['ch'].isin(target_channels)].copy()

    group_data['Power (dB)'] = group_data[alpha_cols].mean(axis=1)
    plot_data = (
        group_data.groupby(['cond', 'ch', 'time'], as_index=False)
        .agg(mean_power=('Power (dB)', 'mean'), sem_power=('Power (dB)', 'sem'))
    )

    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle(f"{ethnicity} - Alpha Power (Fz, Cz, Pz)", fontsize=20)

    for i, ch in enumerate(target_channels):
        ch_data = plot_data[plot_data['ch'] == ch]

        # Row 1: raw condition time courses with SE
        for cond in horder:
            df = ch_data[ch_data['cond'] == cond]
            axs[0, i].plot(df['time'], df['mean_power'], label=cond)
            axs[0, i].fill_between(df['time'],
                                   df['mean_power'] - df['sem_power'],
                                   df['mean_power'] + df['sem_power'],
                                   alpha=0.3)
        axs[0, i].set_title(ch)
        axs[0, i].axvline(0, color='black', linestyle='--')
        axs[0, i].legend()

        # Row 2: Mean diff with SE
        for prefix in ['Mean', 'Vocal']:
            cong_label = f"{prefix}_Cong"
            incong_label = f"{prefix}_Incong"
            cong = ch_data[ch_data['cond'] == cong_label]
            incong = ch_data[ch_data['cond'] == incong_label]
            diff = incong['mean_power'].values - cong['mean_power'].values
            se = np.sqrt(incong['sem_power'].values ** 2 + cong['sem_power'].values ** 2)
            axs[1, i].plot(incong['time'], diff, label=f'{prefix}_Diff')
            axs[1, i].fill_between(incong['time'], diff - se, diff + se, alpha=0.3)
        axs[1, i].axvline(0, color='black', linestyle='--')
        axs[1, i].legend()

        # Row 3: t-test with SE and highlight
        for prefix in ['Mean', 'Vocal']:
            diff_vals = []
            se_vals = []
            times = sorted(ch_data['time'].unique())
            for t in times:
                cong_label = f"{prefix}_Cong"
                incong_label = f"{prefix}_Incong"
                congp = group_data[(group_data['ch'] == ch) & (group_data['cond'] == cong_label) & (group_data['time'] == t)]['Power (dB)']
                incongp = group_data[(group_data['ch'] == ch) & (group_data['cond'] == incong_label) & (group_data['time'] == t)]['Power (dB)']
                if len(incongp) > 0 and len(congp) > 0:
                    tval, pval = ttest_rel(incongp, congp)
                    diff_vals.append(incongp.mean() - congp.mean())
                    se_vals.append(np.sqrt(incongp.sem()**2 + congp.sem()**2))
                    if pval < 0.05:
                        axs[2, i].axvspan(t - 0.01, t + 0.01, color='green', alpha=0.3)
                else:
                    diff_vals.append(np.nan)
                    se_vals.append(np.nan)
            axs[2, i].plot(times, diff_vals, label=f'{prefix}_Diff')
            axs[2, i].fill_between(times, np.array(diff_vals) - np.array(se_vals), np.array(diff_vals) + np.array(se_vals), alpha=0.3)
        axs[2, i].axvline(0, color='black', linestyle='--')
        axs[2, i].legend()

    for ax_row in axs:
        for ax in ax_row:
            ax.set_ylim([-0.3, 0.3])  # adjust according to your data range
            ax.set_xlim([plot_data['time'].min(), plot_data['time'].max()])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Alpha Power (dB)')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{out_dir}/UpperAlpha_ERP_{group}_FzCzPz.png')
    plt.close()


# ----------------------- Plot Time Courses (all channels & all bands) ----------------------- #
# subjinf = pd.read_csv('ParticipantSelfCon.csv')
# idvar = ['cond', 'ch', 'time']
# horder = ['Mean_Cong', 'Mean_Incong', 'Vocal_Cong', 'Vocal_Incong']
#
# for c in ['euro', 'asia']:
#     cc = 'European American' if c == 'euro' else 'East Asian'
#     gid = subjinf.query('`Ethnicity Group`==@cc')['ID'].values
#     snstf = condtf[condtf['subjid'].isin(gid)]
#
#     for band, (start, end) in band_idx.items():
#         cols = [f'band{n}' for n in range(start, end)]
#         reldf = pd.concat([snstf[idvar], snstf[cols]], axis=1)
#         reldf = pd.melt(reldf, id_vars=idvar, value_name='Power (dB)')
#
#         sns.set(font_scale=1.5, style="ticks", rc={"lines.linewidth": 1})
#         g = sns.relplot(x='time', y='Power (dB)', hue='cond', col='ch', kind='line',
#                         data=reldf, facet_kws=dict(sharey=False), aspect=2, height=4, col_wrap=6,
#                         hue_order=horder)
#         g.set_titles(col_template="{col_name}", row_template="{row_name}")
#         for ax in g.axes.flat:
#             ax.axvline(0, color="black", linestyle="--", linewidth=1)
#             ax.axhline(0, color="black", linestyle="--", linewidth=1)
#
#         outname = f'{out_dir}/{band.title()}_Att_VS_{c}_5C.png'
#         os.makedirs(os.path.dirname(outname), exist_ok=True)
#         plt.savefig(outname); plt.close()
