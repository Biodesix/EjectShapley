import pickle
from matplotlib import pyplot as plt
import seaborn as sbn


def draw_fig3_violin(SVs, labels, outfname, delta_mus, n_uninform, used_ftrs):
    nsamples = len(SVs[0])
    nftrs = len(SVs[0][0])
    x_min = -0.01
    x_max = 0.01

    inform_x = []
    inform_y = []
    inform_hue = []

    uninform_x = []
    uninform_y = []
    uninform_hue = []

    for ii in range(len(SVs)):
        for isample in range(nsamples):
            for iftr in range(nftrs):
                if iftr < n_uninform:
                    # uninform
                    uninform_y.append(ii)
                    this_sv = SVs[ii][isample][iftr]
                    if labels[ii][isample] < 0.5:
                        this_sv = this_sv*(-1.0)
                    uninform_x.append(this_sv)
                    if iftr in used_ftrs[ii][isample]:
                        uninform_hue.append('non-null')
                    else:
                        uninform_hue.append('null')
                else:
                    inform_y.append(ii)
                    this_sv = SVs[ii][isample][iftr]
                    if labels[ii][isample] < 0.5:
                        this_sv = this_sv*(-1.0)
                    inform_x.append(this_sv)
                    if iftr in used_ftrs[ii][isample]:
                        inform_hue.append('non-null')
                    else:
                        inform_hue.append('null')

    plt.figure(figsize=(5,8))
    plt.rcParams.update({'font.size': 14})
    plt.xlim(x_min, x_max)
    ax = sbn.violinplot(x=uninform_x, y=uninform_y, hue=uninform_hue, orient='h', inner='quartile', dodge=True)
    for aa in ax.collections:
        aa.set_alpha(0.5)
    for aa in ax.get_legend().legendHandles:
        aa.set_alpha(0.5)
    plt.yticks(rotation=45, fontsize='small')
    ax.set_yticklabels(delta_mus)
    plt.xticks(rotation=45, fontsize='small')
    plt.xlabel('SV~')
    plt.ylabel('Expression Difference')
    plt.tight_layout()
    plt.savefig('./plots/uninform_violin_%s.png'%(outfname), format="png", dpi=1200)
    plt.clf()
    plt.close()

    plt.figure(figsize=(5,8))
    plt.rcParams.update({'font.size': 14})
    plt.xlim(x_min, x_max)
    ax = sbn.violinplot(x=inform_x, y=inform_y, hue=inform_hue, orient='h', inner='quartile', dodge=True)
    for aa in ax.collections:
        aa.set_alpha(0.5)
    for aa in ax.get_legend().legendHandles:
        aa.set_alpha(0.5)
    plt.yticks(rotation=45, fontsize='small')
    ax.set_yticklabels(delta_mus)
    plt.xticks(rotation=45, fontsize='small')
    plt.xlabel('SV~')
    plt.ylabel('Expression Difference')
    plt.tight_layout()
    plt.savefig('./plots/inform_violin_%s.png'%(outfname), format="png", dpi=1200)
    plt.clf()
    plt.close()

if __name__ == '__main__':
    SVs_path = './results/fig3'
    fnames = ['0p25', '0p50', '0p75', '1p0']
    delta_mus = [0.25*(ii+1) for ii in range(len(fnames))]
    n_uninform = 200

    SVs_ts = []
    SVs_ej = []
    labels =[]
    used_ftrs = []
    for name in fnames:
        with open('%s/SVs_%s.p'%(SVs_path, name), 'rb') as infile:
            this_res = pickle.load(infile)
            SVs_ts.append(this_res['SVs_ts'])
            SVs_ej.append(this_res['SVs_ej'])
            labels.append(this_res['labels'])
            used_ftrs.append(this_res['used_ftrs'])

    draw_fig3_violin(SVs_ts, labels, 'ts', delta_mus, n_uninform, used_ftrs)
    draw_fig3_violin(SVs_ej, labels, 'ej', delta_mus, n_uninform, used_ftrs)

    fnames = ['1p25', '1p50', '1p75', '2p0']
    delta_mus = [1.0+0.25*(ii+1) for ii in range(len(fnames))]

    SVs_ts = []
    SVs_ej = []
    labels =[]
    used_ftrs = []
    for name in fnames:
        with open('%s/SVs_%s.p'%(SVs_path, name), 'rb') as infile:
            this_res = pickle.load(infile)
            SVs_ts.append(this_res['SVs_ts'])
            SVs_ej.append(this_res['SVs_ej'])
            labels.append(this_res['labels'])
            used_ftrs.append(this_res['used_ftrs'])

    draw_fig3_violin(SVs_ts, labels, 'ts_high', delta_mus, n_uninform, used_ftrs)
    draw_fig3_violin(SVs_ej, labels, 'ej_high', delta_mus, n_uninform, used_ftrs)