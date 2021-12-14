import crast.sampling
import crast.ctree
import crast.shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
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

    plt.figure(figsize=(5,4))
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
    plt.xlabel('SVs')
    plt.tight_layout()
    plt.savefig(out_path, format="png", dpi=1200)
    plt.clf()
    plt.close()

if __name__ == '__main__':
    print('Welcome to figure E2.  This produces an abbreviated set of data compared to what is in manuscript.  Please see source code for details')
    n_trees = 10 # 1000 trees in manuscript
    train_path = './data/NHANESI_training.csv'
    test_path = './data/NHANESI_validation.csv'
    ftr_names = ['sex_isFemale', 'age', 'physical_activity', 'alkaline_phosphatase', 'SGOT', 'BUN', 'calcium', 'creatinine', 'potassium', 'sodium', 'total_bilirubin', 'red_blood_cells', 'white_blood_cells', 'hemoglobin', 'hematocrit', 'segmented_neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils', 'band_neutrophils', 'cholesterol', 'urine_pH', 'uric_acid', 'systolic_blood_pressure', 'pulse_pressure', 'bmi']
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

    clf = RandomForestClassifier(min_samples_leaf=5, random_state=0, n_estimators=n_trees)
    clf.fit(train_data, train_labels)

    trees = []
    for ee in clf.estimators_:
        trees.append(shap.explainers._tree.SingleTree(ee.tree_, normalize=True))

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

    labels = clf.predict(test_data)
    labels_forest = [0 for ii in range(nsamples)]

    acc = 0
    for ii in range(nsamples):
        if labels[ii] == test_labels[ii]:
            acc += 1
    print('rf accuracy in validation is: %f'%(acc/nsamples))

    labels_new = [0.0 for ii in range(nsamples)]
    SVs_ts = [[0.0 for jj in range(nftrs)] for ii in range(nsamples)]
    SVs_ej = [[0.0 for jj in range(nftrs)] for ii in range(nsamples)]
    new_trees = []
    for it, tt in enumerate(trees):
        this_tree = crast.ctree.Tree.from_sk_shap_tree(tt)
        new_trees.append(this_tree)
        for isample in range(nsamples):
            labels_new[isample] += this_tree.predict_tree(se.data[isample]['features'])

    used_ftrs = [[] for ii in range(nsamples)]
    for it, tt in enumerate(new_trees):
        if it%10 == 0:
            print('on tree %i of %i'%(it, len(new_trees)))
        st = crast.shap.Tree.from_regression_tree(tt)
        st.shap_init()
        bias = 0
        for isample in range(nsamples):
            svs_ts = st.shap_values(se.data[isample]['features'])
            svs_ej = st.shap_values_eject_path_new(se.data[isample]['features'])
            ftr_path = st.get_ftr_path(se.data[isample]['features'])
            for ff in ftr_path:
                if not ff in used_ftrs[isample]:
                    used_ftrs[isample].append(ff)

            for iftr in range(nftrs):
                SVs_ts[isample][iftr] += svs_ts[iftr]
                SVs_ej[isample][iftr] += svs_ej[iftr]

    for isample in range(nsamples):
        for iftr in range(nftrs):
            SVs_ts[isample][iftr] = SVs_ts[isample][iftr]/len(trees)
            SVs_ej[isample][iftr] = SVs_ej[isample][iftr]/len(trees)

    SVs_ts_int = shap.TreeExplainer(clf, data=train_data).shap_values(test_data, check_additivity=False)

    n_null_svs = [0 for ii in range(nsamples)]
    n_null_svs_int = [0 for ii in range(nsamples)]
    ts_res = [0.0 for ii in range(nsamples*nftrs)]
    ts_ej_res = [0.0 for ii in range(nsamples*nftrs)]
    res_idx = 0
    for isample in range(nsamples):
        thresh = abs(sum(SVs_ts[isample]))/nftrs
        for iftr in range(nftrs):
            ts_res[res_idx] = (SVs_ts[isample][iftr] - SVs_ts_int[0][isample][iftr])
            ts_ej_res[res_idx] = (SVs_ts[isample][iftr] - SVs_ej[isample][iftr])
            res_idx += 1
            if not iftr in used_ftrs[isample]:
                if abs(SVs_ts[isample][iftr]) > thresh:
                    n_null_svs[isample] += 1
                if abs(SVs_ts_int[0][isample][iftr]) > thresh:
                    n_null_svs_int[isample] += 1

    print('ts total null SVs above thresh: %i'%(sum(n_null_svs)))
    print('interventional ts total null SVs above thresh: %i'%(sum(n_null_svs_int)))

    from matplotlib import pyplot as plt
    plt.figure()
    plt.hist(n_null_svs) 
    plt.xlabel('Null SV False Positives')
    plt.savefig('./plots/ts_null_FP.png')
    plt.clf()
    plt.close()

    plt.figure()
    plt.hist(n_null_svs_int) 
    plt.xlabel('Null SV False Positives')
    plt.savefig('./plots/ts_int_null_FP.png')
    plt.clf()
    plt.close()

    plt.figure()
    start=-0.03
    stop = 0.03
    nbins = 75
    plt.hist(ts_res, bins=[start + ii*(stop-start)/nbins for ii in range(nbins+1)], label='Interventional', alpha=0.7) 
    plt.hist(ts_ej_res, bins=[start + ii*(stop-start)/nbins for ii in range(nbins+1)], label='Eject', alpha=0.7) 
    plt.xlabel('TreeSHAP SV - (Eject/Interventional) SV')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/ts_res.png', format="png", dpi=1200)
    plt.clf()
    plt.close()


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
    for ii in range(-4,0):
        use_idxs.append(sort_idxs[ii])
    sort_idxs = np.argsort(g2_mean_svs)
    for ii in range(-4,0):
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
    for ii in range(-4,0):
        use_idxs.append(sort_idxs[ii])
    sort_idxs = np.argsort(g2_mean_svs)
    for ii in range(-4,0):
        use_idxs.append(sort_idxs[ii])

    use_idxs = list(set(use_idxs))

    draw_SVs_ts = []
    draw_SVs_ts_int = []
    draw_SVs_ej = []
    for isample in range(nsamples):
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(SVs_ts[isample][iftr])
        draw_SVs_ts.append(these_svs)
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(SVs_ts_int[0][isample][iftr])
        draw_SVs_ts_int.append(these_svs)
        these_svs = []
        for iftr in use_idxs:
            these_svs.append(SVs_ej[isample][iftr])
        draw_SVs_ej.append(these_svs)

    ftr_names = ['sex_isFemale', 'age', 'physical_activity', 'alkaline_phosphatase', 'SGOT', 'BUN', 'calcium', 'creatinine', 'potassium', 'sodium', 'total_bilirubin', 'red_blood_cells', 'white_blood_cells', 'hemoglobin', 'hematocrit', 'segmented_neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils', 'band_neutrophils', 'cholesterol', 'urine_pH', 'uric_acid', 'systolic_blood_pressure', 'pulse_pressure', 'bmi']
    y_labels = []
    for ii in use_idxs:
        y_labels.append(ftr_names[ii])
    x_min = -0.35
    x_max = 0.2
    violin_plot(draw_SVs_ts, labels, './plots/violin_ts.png', x_min=x_min, x_max=x_max, y_labels=y_labels)
    violin_plot(draw_SVs_ts_int, labels, './plots/violin_ts_int.png', x_min=x_min, x_max=x_max, y_labels=y_labels)
    violin_plot(draw_SVs_ej, labels, './plots/violin_ej.png', x_min=x_min, x_max=x_max, y_labels=y_labels)