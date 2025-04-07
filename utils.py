
import os, copy
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import pandas as pd
import mne
from statsmodels.stats.multitest import multipletests

corder = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2',
          'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
          'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']

def butter_filtfilt(data, lowcut, highcut, sr, order=2, mode='band'):
    nyq = 0.5 * sr
    high = highcut / nyq
    if mode == 'band':
        low = lowcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
    elif mode == 'low':
        b, a = signal.butter(order, high, btype='low')
    # y = lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)
    return y

def multicomp(res):
    res = res.assign(pfdr=multipletests(res['p'], method='fdr_bh')[1],
                     pbonf=res['p']*len(res))
    rd = lambda x: round(x, 3)
    res['p'] = rd(res['p']); res['pfdr'] = rd(res['pfdr']); res['pbonf'] = rd(res['pbonf'])
    return res

#######################################################################
# figure
#######################################################################
def get_concat4(im1, im2, im3, im4):
    dst = Image.new('RGB', (im2.width+im3.width, im1.height+im2.height), color=(255, 255, 255))
    ### left columns
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    ### right columns
    dst.paste(im3, (im1.width, 0))
    dst.paste(im4, (im2.width, im1.height))

    return dst


def get_concat2(im1, im2):
    dst = Image.new('RGB', (im1.width+im2.width, im1.height), color=(255, 255, 255))
    ### left columns
    dst.paste(im1, (0, 0))
    ### right columns
    dst.paste(im2, (im1.width, 0))

    return dst

#######################################################
# Topography
#######################################################
def maketopo(dat, head, emocond, tmin=0.4, tmax=1):
    montdat = mne.io.read_raw_brainvision('data/MS001-pre.vhdr')
    # montdat = mne.io.read_raw_brainvision('/Volumes/Experiment/EXP45_MaemukiMeasurement/Dataset/Raw/007_LPP/QST001/MS001-pre.vhdr')
    # montdat = make_montage(montdat)
    info = montdat.info

    if '4emo' in emocond:
        emos = ['posh', 'posl', 'negh', 'negl']
    else:
        emos = ["neg", "pos"]
    dv = 'uV'
    for emo in emos:  #emo='posh'
        topodat = dat.query('emo==@emo').drop('emo', axis=1)
        def topocomp(dat, tmin, tmax, cond='diff'):
            topo = lambda x, y, z: (dat.query('time>@x & time<@y & cond==@z')
                                    .drop('cond', axis=1).groupby(['time', 'loc']).mean())
            topo_pre = topo(tmin, tmax, 'pre'); topo_post = topo(tmin, tmax, 'post')
            topodat = copy.deepcopy(topo_post)
            if cond == 'diff':
                topodat[dv] = topo_post[dv] - topo_pre[dv]
            elif cond == 'pre':
                topodat[dv] = topo_pre[dv]
            else:
                topodat[dv] = topo_post[dv]

            avech = topodat.groupby('loc').mean().reindex(info.ch_names)[dv]
            avech = avech[~avech.index.isin(['FT9', 'FT10', 'TP9'])].fillna(0)
            # avech['Fp1'] = 0; avech['Fp2'] = 0

            return avech

        for cond in ['diff', 'pre', 'post']:
            if cond == 'diff':
                leg = 'Post - Pre'
            elif cond == 'pre':
                leg = 'Pre'
            else:
                leg = 'Post'

            topo = [topocomp(topodat, 0, tmin, cond), topocomp(topodat, tmin, tmax, cond)]
            if cond == 'diff':
                vmin = min([topo[0].min(), topo[1].min()]); vmin = [vmin, vmin]
                vmax = max([topo[0].max(), topo[1].max()]); vmax = [vmax, vmax]

            si = lambda x: str(int(x*1000))
            cap = ['0-'+si(tmin)+' ms', si(tmin)+'-'+si(tmax)+' ms']
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax = [ax1, ax2]
            for i in range(2):  # i=0
                im, cm = mne.viz.plot_topomap(topo[i], pos=info, axes=ax[i], show=False,
                                              vlim=(vmin[i], vmax[i]), cmap='Spectral_r')
                ax[i].set_title(cap[i], fontsize=20)
            # manually fiddle the position of colorbar
            ax_x_start = 0.25; ax_x_width = 0.45
            ax_y_start = 0.1; ax_y_height = 0.08
            cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
            clb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            clb.ax.set_title(emo.title().replace('h', '-High').replace('l', '-Low') +
                             ' [' + leg + ']', fontsize=12, y=-0.1)
            plt.tight_layout()
            plt.savefig(head + 'topomap_' + emo + '_' + cond + '.png'); plt.close()

    # integrate all figures
    getfig = lambda x, y: Image.open(head + 'topomap_' + x + '_' + y + '.png')
    delfig = lambda x, y: os.remove(head + 'topomap_' + x + '_' + y + '.png')
    for tt in ['pre', 'post', 'diff']:
        if '4emo' in emocond:
            im1 = getfig('negh', tt); im2 = getfig('negl', tt);
            im3 = getfig('posh', tt); im4 = getfig('posl', tt)
            get_concat4(im1, im2, im3, im4).save(head + 'topomap_' + emocond + '_' + tt + '.png')
            delfig('negh', tt); delfig('negl', tt);
            delfig('posh', tt); delfig('posl', tt)
        else:
            im1 = getfig('neg', tt); im2 = getfig('pos', tt)
            get_concat2(im1, im2).save(head + 'topomap_' + emocond + '_' + tt + '.png')
            delfig('neg', tt); delfig('pos', tt)
        plt.close()
