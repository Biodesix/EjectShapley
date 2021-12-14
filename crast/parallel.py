import crast.sampling
import crast.ctree
import crast.shap
import pickle


# TODO: need to set seed on sampling engine

def train_SV_forest(nbags, job_id, se, se_val, ftr_names, save_path):
    se.set_seed(job_id)

    forest = []
    for ibag in range(nbags):
        se.generate_sampling(0.667)

        ct = crast.ctree.Tree()
        ct.read_from_sampling_engine(se.get_training_samples())
        ct.bin_continuous_features(10)
        ct.fit()

        forest.append(ct)

    test_set = []
    test_set_defs = []

    for ss in se_val.data:
        test_set.append(ss['features'])
        test_set_defs.append(int(ss['class']))

    nsamples = len(test_set)
    nfeatures = len(ftr_names)

    labels = [[-999 for ii in range(nbags)] for ii in range(nsamples)]
    for ii in range(nsamples):
        this_label = 0
        for itree, tree in enumerate(forest):
            labels[ii][ibag] = int(tree.predict_tree(test_set[ii]))

    SVs_ts = [[[0 for kk in range(nbags)] for jj in range(nfeatures)] for ii in range(nsamples)]
    SVs_ej = [[[0 for kk in range(nbags)] for jj in range(nfeatures)] for ii in range(nsamples)]

    for itree, tree in enumerate(forest):
        if itree%10 == 0:
            print(itree)
        st = crast.shap.Tree.from_tree(tree)
        st.shap_init()

        bias = 0
        for it, tt in enumerate(test_set):
            svs_ts = st.shap_values(tt)
            # get bias and correct it
            if it == 0:
                if sum(svs_ts) < 0:
                    bias = sum(svs_ts) + 1
                else:
                    bias = sum(svs_ts) - 1

            for ii in range(len(svs_ts)):
                svs_ts[ii] -= bias/len(svs_ts)
            
            svs_ej = st.shap_values_eject_path(tt)

            for iftr in range(len(ftr_names)):
                SVs_ts[it][iftr][itree] = svs_ts[iftr]
                SVs_ej[it][iftr][itree] = svs_ej[iftr]

    out_res = {}
    out_res['SVs_ts'] = SVs_ts
    out_res['SVs_ej'] = SVs_ej
    out_res['forest'] = forest
    out_res['labels'] = labels
    with open('%s/SVs_%i.p'%(save_path, job_id)) as ofile:
        pickle.dump(out_res, ofile)