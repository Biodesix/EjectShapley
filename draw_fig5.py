import xgboost as xgb
import crast.sampling
import numpy as np
import crast.ctree
import crast.shap
from matplotlib import pyplot as plt
import seaborn as sbn

def draw_fig5GH(SVs_ts, SVs_ej, labels, ftr_names):
    ftr_names_plot = []
    for ff in ftr_names:
        ftr_names_plot.append(ff.replace('_', ' '))

    nsamples = len(labels)
    nftrs = len(ftr_names)

    test_path = './data/NHANESI_validation.csv'
    se = crast.sampling.Engine()
    se.read_data(test_path, ftr_names, 'SampleID', class_name='death')

    for iftr, ftr in enumerate(ftr_names):
        ftr_vals = [-999.9 for ii in range(nsamples)]
        svs_ts = [-999.9 for ii in range(nsamples)]
        svs_ej = [-999.9 for ii in range(nsamples)]
        for isample in range(nsamples):
            ftr_vals[isample] = se.data[isample]['features'][iftr]
            svs_ts[isample] = SVs_ts[isample][iftr]
            svs_ej[isample] = SVs_ej[isample][iftr]

        plt.figure()
        plt.rcParams.update({'font.size': 14})
        plt.scatter(ftr_vals, svs_ts, color='r', label='TreeShap', alpha=0.5)
        plt.scatter(ftr_vals, svs_ej, color='b', label='Eject', alpha=0.5)
        plt.xlabel(ftr_names_plot[iftr])
        plt.ylabel('Shapley Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./plots/SVs_x_%s.png'%(ftr), format="png", dpi=1200)
        plt.clf()
        plt.close()

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
    train_path = './data/NHANESI_training.csv'
    test_path = './data/NHANESI_validation.csv'
    ftr_names = ['sex_isFemale', 'age', 'physical_activity', 'alkaline_phosphatase', 'SGOT', 'BUN', 'calcium', 'creatinine', 'potassium', 'sodium', 'total_bilirubin', 'red_blood_cells', 'white_blood_cells', 'hemoglobin', 'hematocrit', 'segmented_neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils', 'band_neutrophils', 'cholesterol', 'urine_pH', 'uric_acid', 'systolic_blood_pressure', 'pulse_pressure', 'bmi']
    nbags = 625
    se = crast.sampling.Engine()
    se.read_data(train_path, ftr_names, 'SampleID', class_name='death')

    nsamples = len(se.data)
    nftrs = len(se.data[0]['features'])

    train_data = [[-999.9 for jj in range(nftrs)] for ii in range(nsamples)]
    train_labels = [-999 for ii in range(nsamples)]
    for isample in range(nsamples):
        train_labels[isample] = int(se.data[isample]['class'])
        for iftr in range(nftrs):
            train_data[isample][iftr] = se.data[isample]['features'][iftr]

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    bdt = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5
    )
    bdt.fit(train_data, train_labels)

    se = crast.sampling.Engine()
    se.read_data(test_path, ftr_names, 'SampleID', class_name='death')

    nsamples = len(se.data)
    nftrs = len(se.data[0]['features'])

    test_data = [[-999.9 for jj in range(nftrs)] for ii in range(nsamples)]
    test_labels = [-999 for ii in range(nsamples)]
    for isample in range(nsamples):
        test_labels[isample] = int(se.data[isample]['class'])
        for iftr in range(nftrs):
            test_data[isample][iftr] = se.data[isample]['features'][iftr]

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    labels = bdt.predict(test_data)
    labels_forest = [0 for ii in range(nsamples)]

    acc = 0
    for ii in range(nsamples):
        if labels[ii] == test_labels[ii]:
            acc += 1
    print('bdt accuracy in validation is: %f'%(acc/nsamples))

    booster = bdt.get_booster()
    import shap
    shap_load = shap.explainers._tree.XGBTreeModelLoader(booster)
    trees = shap_load.get_trees()
    labels_new = [0.0 for ii in range(nsamples)]

    SVs_ts = [[0.0 for jj in range(nftrs)] for ii in range(nsamples)]
    SVs_ej = [[0.0 for jj in range(nftrs)] for ii in range(nsamples)]
    new_trees = []
    for it, tt in enumerate(trees):
        this_tree = crast.ctree.Tree.from_shap_tree(tt)
        new_trees.append(this_tree)
        for isample in range(nsamples):
            labels_new[isample] += this_tree.predict_tree(se.data[isample]['features'])

    for tt in new_trees:
        st = crast.shap.Tree.from_regression_tree(tt)
        st.shap_init()
        bias = 0
        for isample in range(nsamples):
            svs_ts = st.shap_values(se.data[isample]['features'])
            svs_ej = st.shap_values_eject_path_new(se.data[isample]['features'])

            for iftr in range(nftrs):
                SVs_ts[isample][iftr] += svs_ts[iftr]
                SVs_ej[isample][iftr] += svs_ej[iftr]

    import math
    for isample in range(nsamples):
        if (1.0/(1.0+math.exp(-1.0*labels_new[isample]))) < 0.5:
            labels_new[isample] = 0
        else:
            labels_new[isample] = 1

    acc = 0
    for ii in range(nsamples):
        if labels[ii] == labels_new[ii]:
            acc += 1
    print('bdt label recovery is: %f'%(acc/nsamples))

    draw_fig5GH(SVs_ts, SVs_ej, labels_new, ftr_names)

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

    draw_SVs_ts = []
    draw_SVs_ej = []
    for isample in range(nsamples):
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(SVs_ts[isample][iftr])
        draw_SVs_ts.append(these_svs)
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(SVs_ej[isample][iftr])
        draw_SVs_ej.append(these_svs)

    ftr_names = ['sex_isFemale', 'age', 'physical_activity', 'alkaline_phosphatase', 'SGOT', 'BUN', 'calcium', 'creatinine', 'potassium', 'sodium', 'total_bilirubin', 'red_blood_cells', 'white_blood_cells', 'hemoglobin', 'hematocrit', 'segmented_neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils', 'band_neutrophils', 'cholesterol', 'urine_pH', 'uric_acid', 'systolic_blood_pressure', 'pulse_pressure', 'bmi']
    y_labels = []
    for ii in use_idxs:
        y_labels.append(ftr_names[ii])
    violin_plot(draw_SVs_ts, labels, './plots/violin_ts.png', x_min=-6, x_max=6, y_labels=y_labels)
    violin_plot(draw_SVs_ej, labels, './plots/violin_ej.png', x_min=-6, x_max=6, y_labels=y_labels)

    mean_abs_svs = [0.0 for ii in range(len(use_idxs))]
    for ii in range(len(use_idxs)):
        for isample in range(nsamples):
            mean_abs_svs[ii] += abs(SVs_ts[isample][use_idxs[ii]])
    ftr_sort_idxs = [use_idxs[ii] for ii in np.argsort(mean_abs_svs)]

    g1_idxs = []
    g1_largest_svs = []
    g2_idxs = []
    g2_largest_svs = []
    for isample in range(nsamples):
        if labels[isample] > 0.5:
            g1_idxs.append(isample)
            g1_largest_svs.append(SVs_ts[isample][ftr_sort_idxs[-1]])
        else:
            g2_idxs.append(isample)
            g2_largest_svs.append(-1*SVs_ts[isample][ftr_sort_idxs[-1]])

    g1_sort_idxs = np.argsort(g1_largest_svs)
    g2_sort_idxs = np.argsort(g2_largest_svs)

    sample_sort_idxs = [g1_idxs[idx] for idx in g1_sort_idxs]

    draw_SVs_ts = [[SVs_ts[isample][iftr] for iftr in ftr_sort_idxs] for isample in sample_sort_idxs]
    draw_SVs_ej = [[SVs_ej[isample][iftr] for iftr in ftr_sort_idxs] for isample in sample_sort_idxs]
    draw_SVs_diff = [[(SVs_ts[isample][iftr] - SVs_ej[isample][iftr]) for iftr in ftr_sort_idxs] for isample in sample_sort_idxs]

    from matplotlib import pyplot as plt
    plt.imshow(draw_SVs_ts, aspect='auto')
    plt.clim(-7, 7)
    cb = plt.colorbar()
    cb.set_label('SV')
    plt.xticks([ii for ii in range(len(ftr_sort_idxs))], labels=[' ' for ii in ftr_sort_idxs], rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/heatmap_ts.png')
    plt.clf()
    plt.close()

    plt.imshow(draw_SVs_ej, aspect='auto')
    plt.clim(-7, 7)
    plt.colorbar()
    cb.set_label('SV')
    plt.xticks([ii for ii in range(len(ftr_sort_idxs))], labels=[' ' for ii in ftr_sort_idxs], rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/heatmap_ej.png')
    plt.clf()
    plt.close()

    plt.imshow(draw_SVs_diff, aspect='auto')
    plt.clim(-3.5, 3.5)
    plt.colorbar()
    cb.set_label('TreeShap - Eject')
    plt.xticks([ii for ii in range(len(ftr_sort_idxs))], labels=[ftr_names[ii] for ii in ftr_sort_idxs], rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/heatmap_diff.png')
    plt.clf()
    plt.close()

    draw_samples = [5, 30, 55]
    for isample in draw_samples:
        plt.figure()
        plt.scatter([ii for ii in range(len(ftr_sort_idxs))], draw_SVs_ts[isample], label='TreeShap')
        plt.scatter([ii for ii in range(len(ftr_sort_idxs))], draw_SVs_ej[isample], label='Eject')
        plt.legend()
        plt.axhline(y=sum(SVs_ej[sample_sort_idxs[isample]])/nftrs, color='black', linestyle='-')
        plt.axhline(y=-1*sum(SVs_ej[sample_sort_idxs[isample]])/nftrs, color='black', linestyle='-')
        plt.xticks([ii for ii in range(len(ftr_sort_idxs))], labels=[ftr_names[ii] for ii in ftr_sort_idxs], rotation=45)
        plt.tight_layout()
        plt.savefig('./plots/draw_samples_SVs_%i.png'%(isample))
        plt.clf()
        plt.close()


    ftr_names = ['gender', 'age', 'phys. activity', 'a. phosphatase', 'SGOT', 'BUN', 'calcium', 'creatinine', 'potassium', 'sodium', 'bilirubin', 'red_blood_cells', 'w. blood cells', 'hemoglobin', 'hematocrit', 'segmented_neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils', 'band_neutrophils', 'cholesterol', 'urine_pH', 'uric_acid', 'systolic b.p.', 'pulse_pressure', 'bmi']
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6,12))

    im = ax[0].imshow(draw_SVs_ts, aspect='auto', vmin=-7, vmax=7)
    cb = plt.colorbar(im, ax=ax[0])
    cb.set_label('TreeShap SV', fontsize=14)
    ax[0].set_xticks([ii for ii in range(len(ftr_sort_idxs))])

    im = ax[1].imshow(draw_SVs_ej, aspect='auto', vmin=-7, vmax=7)
    cb = plt.colorbar(im, ax=ax[1])
    cb.set_label('Eject SV', fontsize=14)
    ax[1].set_xticks([ii for ii in range(len(ftr_sort_idxs))])

    im = ax[2].imshow(draw_SVs_diff, aspect='auto', vmin=-3.5, vmax=3.5)
    cb = plt.colorbar(im, ax=ax[2])
    cb.set_label('TreeShap SV - Eject SV', fontsize=14)
    ax[2].set_xticks([ii for ii in range(len(ftr_sort_idxs))])

    ax[0].set_xticklabels([ftr_names[ii] for ii in ftr_sort_idxs], rotation = 45, ha="right")
    ax[1].set_xticklabels([ftr_names[ii] for ii in ftr_sort_idxs], rotation = 45, ha="right")
    ax[2].set_xticklabels([ftr_names[ii] for ii in ftr_sort_idxs], rotation = 45, ha="right")

    ax[0].set_ylabel('Patient', fontsize=14)
    ax[1].set_ylabel('Patient', fontsize=14)
    ax[2].set_ylabel('Patient', fontsize=14)

    plt.tight_layout()
    fig.savefig('./plots/heatmap_overlay.png', format="png", dpi=1200)


    fig, ax = plt.subplots(nrows=len(draw_samples), ncols=1, sharex=True, figsize=(6,4*len(draw_samples)))

    for idraw, isample in enumerate(draw_samples):
        ax[idraw].scatter([ii for ii in range(len(ftr_sort_idxs))], draw_SVs_ts[isample], label='TreeShap')
        ax[idraw].scatter([ii for ii in range(len(ftr_sort_idxs))], draw_SVs_ej[isample], label='Eject')
        ax[idraw].set_xticks([ii for ii in range(len(ftr_sort_idxs))])
        if idraw == 0:
            ax[idraw].legend()
        ax[idraw].axhline(y=sum(SVs_ej[sample_sort_idxs[isample]])/nftrs, color='black', linestyle='-')
        ax[idraw].axhline(y=-1*sum(SVs_ej[sample_sort_idxs[isample]])/nftrs, color='black', linestyle='-')
        ax[idraw].set_xticklabels([ftr_names[ii] for ii in ftr_sort_idxs], rotation = 45, ha="right")
        if idraw == 2:
            ax[idraw].yaxis.set_label_coords(-0.085, 0.5)
        ax[idraw].set_ylabel('Patient %i SV'%(isample), fontsize=14)

    plt.tight_layout()
    fig.savefig('./plots/samples_overlay.png', format="png", dpi=1200)