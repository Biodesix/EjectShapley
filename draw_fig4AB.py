import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

def violin_plot(SVs, labels, out_path, x_min = -1, x_max = 1, y_labels = []):
    nsamples = len(SVs)
    nftrs = len(SVs[0])
    g1_idxs = []
    g2_idxs = []
    for isample in range(nsamples):
        if labels[isample] < 0.5:
            g1_idxs.append(isample)
        else:
            g2_idxs.append(isample)

    g1_data = []
    g2_data = []
    for iftr in range(nftrs):
        this_ftr = []
        for isample in g1_idxs:
            this_ftr.append(SVs[isample][iftr])
        g1_data.append(this_ftr)
        this_ftr = []
        for isample in g2_idxs:
            this_ftr.append(SVs[isample][iftr])
        g2_data.append(this_ftr)

    plt.figure(figsize=(7,11))
    plt.rcParams.update({'font.size': 14})
    plt.xlim(x_min, x_max)
    sbn_x = []
    sbn_y = []
    sbn_hue = []
    for ii, ss1 in enumerate(SVs):
        for jj, ss in enumerate(ss1):
            sbn_x.append(ss)
            sbn_y.append(jj)
            sbn_hue.append(labels[ii])
    ax = sbn.violinplot(x=sbn_x, y=sbn_y, hue=sbn_hue, orient='h', inner='stick', dodge=True)
    for aa in ax.collections:
        aa.set_alpha(0.5)
    for aa in ax.get_legend().legendHandles:
        aa.set_alpha(0.5)
    if len(y_labels) > 0:
        plt.yticks(rotation=45, fontsize='small')
        ax.set_yticklabels(y_labels)
    plt.xticks(rotation=45, fontsize='small')
    plt.xlabel('SVs')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=1200)
    plt.clf()
    plt.close()

if __name__ == '__main__':

    with open('./results/fig4/NKI_dev_combined_full.p', 'rb') as infile:
        this_res = pickle.load(infile)
    SVs_ts = this_res['SVs_ts']
    SVs_ej = this_res['SVs_ej']
    nsamples = len(SVs_ts)
    nftrs = len(SVs_ts[0])

    labels = []
    for ss in SVs_ej:
        if sum(ss) > 0:
            labels.append(1)
        else:
            labels.append(0)

    g1_mean_svs = [0 for ii in range(nftrs)]
    g2_mean_svs = [0 for ii in range(nftrs)]
    for iftr in range(nftrs):
        for isample in range(nsamples):
            if labels[isample] == 1:
                g1_mean_svs[iftr] += SVs_ts[isample][iftr]
            else:
                g2_mean_svs[iftr] += SVs_ts[isample][iftr]
        g1_mean_svs[iftr] = abs(g1_mean_svs[iftr]/nsamples)
        g2_mean_svs[iftr] = abs(g2_mean_svs[iftr]/nsamples)

    use_idxs = []
    sort_idxs = np.argsort(g1_mean_svs)
    for ii in range(-5,0):
        use_idxs.append(sort_idxs[ii])
    sort_idxs = np.argsort(g2_mean_svs)
    for ii in range(-5,0):
        use_idxs.append(sort_idxs[ii])

    g1_mean_svs = [0 for ii in range(nftrs)]
    g2_mean_svs = [0 for ii in range(nftrs)]
    for iftr in range(nftrs):
        for isample in range(nsamples):
            if labels[isample] == 1:
                g1_mean_svs[iftr] += SVs_ej[isample][iftr]
            else:
                g2_mean_svs[iftr] += SVs_ej[isample][iftr]
        g1_mean_svs[iftr] = abs(g1_mean_svs[iftr]/nsamples)
        g2_mean_svs[iftr] = abs(g2_mean_svs[iftr]/nsamples)

    sort_idxs = np.argsort(g1_mean_svs)
    for ii in range(-5,0):
        use_idxs.append(sort_idxs[ii])
    sort_idxs = np.argsort(g2_mean_svs)
    for ii in range(-5,0):
        use_idxs.append(sort_idxs[ii])

    use_idxs = list(set(use_idxs))

    SVs_ts = []
    SVs_ej = []
    for isample in range(nsamples):
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(this_res['SVs_ts'][isample][iftr])
        SVs_ts.append(these_svs)
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(this_res['SVs_ej'][isample][iftr])
        SVs_ej.append(these_svs)

    ftr_names = [str(ii) for ii in range(nftrs)]
    y_labels = []
    for ii in use_idxs:
        y_labels.append(ftr_names[ii])
    violin_plot(SVs_ts, labels, './plots/violin_ts.png', x_min=-0.01, x_max=0.008, y_labels=y_labels)
    violin_plot(SVs_ej, labels, './plots/violin_ej.png', x_min=-0.01, x_max=0.008, y_labels=y_labels)