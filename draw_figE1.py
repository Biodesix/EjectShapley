import pickle
import os
import math

in_dir = './results/figE1/uncorr'
# in_dir = './results/figE1/corr'

nfiles = 0
for ifile, fname in enumerate(os.listdir(in_dir)):
    with open(in_dir + '/' + fname, 'rb') as infile:
        res = pickle.load(infile)
        nfiles += 1
        if ifile == 0:
            SVs_ts = res['SVs_ts']
            SVs_ts_int = res['SVs_ts_int']
            SVs_ej = res['SVs_ej']
            nsamples = len(SVs_ts)
            nftrs = len(SVs_ts[0])
        else:
            for isample in range(nsamples):
                for iftr in range(nftrs):
                    SVs_ts[isample][iftr] += res['SVs_ts'][isample][iftr]
                    SVs_ts_int[isample][iftr] += res['SVs_ts_int'][isample][iftr]
                    SVs_ej[isample][iftr] += res['SVs_ej'][isample][iftr]

for isample in range(nsamples):
    for iftr in range(nftrs):
        SVs_ts[isample][iftr] = SVs_ts[isample][iftr]/nfiles
        SVs_ts_int[isample][iftr] = SVs_ts_int[isample][iftr]/nfiles
        SVs_ej[isample][iftr] = SVs_ej[isample][iftr]/nfiles

for isample in range(nsamples):
    if isample == 0:
        bias = sum(SVs_ts[isample])-sum(SVs_ej[isample])
    for iftr in range(nftrs):
        SVs_ts[isample][iftr] = SVs_ts[isample][iftr] - bias/nftrs

labels = []
ts_res = []
ts_ej_res = []
for isample in range(nsamples):
    if isample == 0:
        bias = sum(SVs_ts[isample]) - sum(SVs_ts_int[isample])
    for iftr in range(nftrs):
        SVs_ts_int[isample][iftr] += bias/nftrs
        ts_res.append(SVs_ts[isample][iftr] - SVs_ts_int[isample][iftr])
        ts_ej_res.append(SVs_ts[isample][iftr] - SVs_ej[isample][iftr])
    if sum(SVs_ej[isample]) > 0:
        labels.append(1)
    else:
        labels.append(0)

from matplotlib import pyplot as plt
plt.figure()
start=-0.07
stop = 0.07
nbins = 25
plt.hist(ts_res, bins=[start + ii*(stop-start)/nbins for ii in range(nbins+1)], label='Interventional', alpha=0.7) 
plt.hist(ts_ej_res, bins=[start + ii*(stop-start)/nbins for ii in range(nbins+1)], label='Eject', alpha=0.7) 
plt.xlabel('TreeSHAP SV - (Eject/Interventional) SV')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/ts_res.png', format="png", dpi=1200)
plt.clf()
plt.close()

import statistics
g1_idxs = []
g2_idxs = []
for isample in range(nsamples):
    if labels[isample] < 0.5:
        g1_idxs.append(isample)
    else:
        g2_idxs.append(isample)

ave_ts = [0 for ii in range(nftrs)]
ave_ej = [0 for ii in range(nftrs)]
med_ts = [[] for ii in range(nftrs)]
med_ej = [[] for ii in range(nftrs)]

g1_ave_ts = [0 for ii in range(nftrs)]
g1_ave_ej = [0 for ii in range(nftrs)]
g1_med_ts = [[] for ii in range(nftrs)]
g1_med_ej = [[] for ii in range(nftrs)]
for isample in g1_idxs:
    for iftr in range(nftrs):
        ave_ts[iftr] += -1*SVs_ts[isample][iftr]
        ave_ej[iftr] += -1*SVs_ej[isample][iftr]

        med_ts[iftr].append(-1*SVs_ts[isample][iftr])
        med_ej[iftr].append(-1*SVs_ej[isample][iftr])

        g1_ave_ts[iftr] += SVs_ts[isample][iftr]
        g1_ave_ej[iftr] += SVs_ej[isample][iftr]

        g1_med_ts[iftr].append(SVs_ts[isample][iftr])
        g1_med_ej[iftr].append(SVs_ej[isample][iftr])

g2_ave_ts = [0 for ii in range(nftrs)]
g2_ave_ej = [0 for ii in range(nftrs)]
g2_med_ts = [[] for ii in range(nftrs)]
g2_med_ej = [[] for ii in range(nftrs)]
for isample in g2_idxs:
    for iftr in range(nftrs):
        ave_ts[iftr] += SVs_ts[isample][iftr]
        ave_ej[iftr] += SVs_ej[isample][iftr]

        med_ts[iftr].append(SVs_ts[isample][iftr])
        med_ej[iftr].append(SVs_ej[isample][iftr])

        g2_ave_ts[iftr] += SVs_ts[isample][iftr]
        g2_ave_ej[iftr] += SVs_ej[isample][iftr]

        g2_med_ts[iftr].append(SVs_ts[isample][iftr])
        g2_med_ej[iftr].append(SVs_ej[isample][iftr])

std_ts = [0 for ii in range(nftrs)]
std_ej = [0 for ii in range(nftrs)]
for iftr in range(nftrs):
    ave_ts[iftr] = ave_ts[iftr]/nsamples
    ave_ej[iftr] = ave_ej[iftr]/nsamples
    std_ts[iftr] = statistics.stdev(med_ts[iftr])/math.sqrt(len(med_ts[iftr]))
    std_ej[iftr] = statistics.stdev(med_ej[iftr])/math.sqrt(len(med_ej[iftr]))
    med_ts[iftr] = statistics.median(med_ts[iftr])
    med_ej[iftr] = statistics.median(med_ej[iftr])

    g1_ave_ts[iftr] = g1_ave_ts[iftr]/len(g1_idxs)
    g1_ave_ej[iftr] = g1_ave_ej[iftr]/len(g1_idxs)
    g1_med_ts[iftr] = statistics.median(g1_med_ts[iftr])
    g1_med_ej[iftr] = statistics.median(g1_med_ej[iftr])

    g2_ave_ts[iftr] = g2_ave_ts[iftr]/len(g2_idxs)
    g2_ave_ej[iftr] = g2_ave_ej[iftr]/len(g2_idxs)
    g2_med_ts[iftr] = statistics.median(g2_med_ts[iftr])
    g2_med_ej[iftr] = statistics.median(g2_med_ej[iftr])

delta_mus = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0]

plt.figure()
plt.ylim(-0.015, 0.2)
plt.errorbar(delta_mus, ave_ts, yerr=std_ts, fmt='o', capsize=5 ,color='r', label='TreeSHAP')
plt.errorbar(delta_mus, ave_ej, yerr=std_ej, fmt='o', capsize=5, color='b', label='Eject')
plt.xlabel('Expression Difference')
plt.ylabel('Group Mean SV~')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/fig2_overlay_mean.png', format="png", dpi=1200)
plt.clf()