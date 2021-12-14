import pickle
from matplotlib import pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches

with open('./results/fig4/NKI_dev_combined_full.p', 'rb') as infile:
    this_res = pickle.load(infile)
    SVs_ts = this_res['SVs_ts']
    used_ftrs = this_res['used_ftrs']

nsamples = len(SVs_ts)
nftrs = len(SVs_ts[0])

null_devs = [0.0 for ii in range(nsamples)]
mean_SVs = [0.0 for ii in range(nsamples)]
n_devs = [0 for ii in range(nsamples)]
dev_hist = []
n_FP = [0 for ii in range(nsamples)]
for isample in range(nsamples):
    mean_abs_sv = 0
    for iftr in range(nftrs):
        mean_abs_sv += abs(SVs_ts[isample][iftr])
    mean_abs_sv = mean_abs_sv/nftrs
    P_thresh = abs(sum(SVs_ts[isample])/nftrs)
    for iftr in range(nftrs):
        mean_SVs[isample] += abs(SVs_ts[isample][iftr])
        if not iftr in used_ftrs[isample]:
            null_devs[isample] += abs(SVs_ts[isample][iftr])
            n_devs[isample] += 1
            dev_hist.append(abs(SVs_ts[isample][iftr])/mean_abs_sv)
            if abs(SVs_ts[isample][iftr]) > P_thresh:
                n_FP[isample] += 1

sort_idxs = np.argsort(null_devs)
for isample in sort_idxs[-10:]:
    mean_pos = 0
    mean_neg = 0
    temp_SVs = []
    for iftr in range(nftrs):
        if SVs_ts[isample][iftr] < 0:
            mean_neg += SVs_ts[isample][iftr]
        else:
            mean_pos += SVs_ts[isample][iftr]
        if not iftr in used_ftrs[isample]:
            temp_SVs.append(SVs_ts[isample][iftr])
    mean_neg = mean_neg/nftrs
    mean_pos = mean_pos/nftrs
    sort_SVs = np.sort(temp_SVs).tolist()
    temp_SVs = sort_SVs[:10]
    for sv in sort_SVs[-10:]:
        temp_SVs.append(sv)
    
    colors = [[255, 0, 0] for ii in range(len(temp_SVs))]
    rand_idxs = random.sample([ii for ii in range(nftrs)], 20)
    for idx in rand_idxs:
        temp_SVs.append(SVs_ts[isample][idx])
        colors.append([0, 0, 0])
    rand_idxs = random.sample([ii for ii in range(len(temp_SVs))], len(temp_SVs))
    draw_SVs = []
    draw_colors = []
    for idx in rand_idxs:
        draw_SVs.append(temp_SVs[idx])
        draw_colors.append(colors[idx])
    plt.figure()
    plt.scatter([ii for ii in range(len(draw_SVs))], draw_SVs, c=np.array(draw_colors)/255.0)
    plt.axhline(y=mean_pos, linestyle='-', color='black')
    plt.axhline(y=mean_neg, linestyle='-', color='black')
    plt.xticks([ii*5 for ii in range(int(len(draw_SVs)/5))], rotation=45)
    red_patch = mpatches.Patch(color='red', label='Null')
    black_patch = mpatches.Patch(color='black', label='Non-null')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.legend(handles=[red_patch, black_patch])
    plt.ylabel('SV')
    plt.xlabel('Feature ID (arbitrary)')
    plt.tight_layout()
    plt.savefig('./plots/max_draw_samples_SVs_%i.png'%(isample), format="png", dpi=1200)
    plt.clf()
    plt.close()

plt.figure()
plt.hist(dev_hist, bins=[0.05*(ii+1) for ii in range(30)]) 
plt.xlabel('Null SV / mean abs SV')
plt.tight_layout()
plt.savefig('./plots/NKI_ts_null_dev.png', format="png", dpi=1200)
plt.clf()
plt.close()

plt.figure()
plt.hist(n_FP) 
plt.xlabel('Null SV False Positives')
plt.tight_layout()
plt.savefig('./plots/NKI_ts_null_FP.png', format="png", dpi=1200)
plt.clf()
plt.close()