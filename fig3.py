import multiprocessing
import crast.sampling
import crast.ctree
import crast.shap
import pickle
import random

def calc_SVs(ftr_names, nbags, train_path, test_path, save_path, job_name):
    se = crast.sampling.Engine()
    se.read_data(train_path, ftr_names, 'SampleID', class_name='Definition')
    forest = []
    for ibag in range(nbags):
        se.generate_sampling(0.667)

        ct = crast.ctree.Tree()
        ct.read_from_sampling_engine(se.get_training_samples())
        ct.bin_continuous_features(10)
        ct.fit()
        # ct.use_binary_leaf_score('1')

        forest.append(ct)

    se_val = crast.sampling.Engine()
    se_val.read_data(test_path, ftr_names, 'SampleID', class_name='Definition')
    test_set = []
    test_set_defs = []

    for ss in se_val.data:
        test_set.append(ss['features'])
        test_set_defs.append(int(ss['class']))

    nsamples = len(test_set)
    nfeatures = len(ftr_names)

    labels = [-999 for ii in range(nsamples)]
    probs = [0 for ii in range(nsamples)]
    for ii in range(nsamples):
        this_label = 0
        for tree in forest:
            this_label += float(tree.predict_tree(test_set[ii]))
        probs[ii] = this_label/len(forest)
        if probs[ii] == 0.5:
            temp = random.sample([0,1], 1)
            labels[ii] = temp[0]
        elif probs[ii] < 0.5:
            labels[ii] = 0
        else:
            labels[ii] = 1

    acc = 0
    for ii in range(nsamples):
        if labels[ii] == test_set_defs[ii]:
            acc += 1/nsamples
    print('job: %s acc: %f'%(job_name, acc))

    SVs_ts = [[0 for jj in range(nfeatures)] for ii in range(nsamples)]
    SVs_ej = [[0 for jj in range(nfeatures)] for ii in range(nsamples)]
    used_ftrs = [[] for ii in range(nsamples)]

    for itree, tree in enumerate(forest):
        if itree%100 == 0:
            print('job: %s on bag %i'%(job_name, itree))
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
            
            svs_ej = st.shap_values_eject_path_new(tt)
            for ftr in st.get_ftr_path(tt):
                if not ftr in used_ftrs[it]:
                    used_ftrs[it].append(ftr)

            for iftr in range(len(ftr_names)):
                SVs_ts[it][iftr] += svs_ts[iftr]/len(forest)
                SVs_ej[it][iftr] += svs_ej[iftr]/len(forest)
    out_res = {}
    out_res['SVs_ts'] = SVs_ts
    out_res['SVs_ej'] = SVs_ej
    out_res['used_ftrs'] = used_ftrs
    out_res['labels'] = labels
    with open('%s\\SVs_%s.p'%(save_path, job_name), 'wb') as ofile:
        pickle.dump(out_res, ofile)

# class for carrying parameters to worker
class mp_worker_input:
    def __init__(self, name, job_id):
        self.name = name
        self.job_id = job_id
        self.nbags = 0
        self.ftr_names = []
        self.save_path = ''
        self.train_path = ''
        self.test_path = ''

# this is the worker, it is where the code that needs executing goes
def mp_worker(job_input):
    print('Job %s submitted'%(job_input.name))
    calc_SVs(job_input.ftr_names, job_input.nbags, job_input.train_path, job_input.test_path, job_input.save_path, job_input.name)
    print('Process %s with id %i\tDONE'%(job_input.name, job_input.job_id)) # should pass something printable for tracking when jobs finish

# call mp_handler() when we run as a script (python //<path>/test_batch.py) to kick off jobs and report progress
if __name__ == '__main__':
    print('Welcome to figure 3.  This produces an abbreviated set of data compared to what is in manuscript.  Please see source code for details')
    nbags = 10 # manuscript used 1000
    save_path = './results/fig3_test'

    ftr_names = []
    for ii in range(1, 201):
        ftr_names.append('U%i'%(ii))
    for ii in range(201, 401):
        ftr_names.append('IU%i'%(ii))

    fnames = ['0p25', '0p50', '0p75', '1p0'] 

    jobs_to_do = []
    for ii, name in enumerate(fnames):
        this_mp = mp_worker_input(name, ii)
        this_mp.nbags = nbags
        this_mp.train_path = './data/fig3/train_%s.csv'%(name)
        this_mp.test_path = './data/fig3/test_%s.csv'%(name)
        this_mp.ftr_names = ftr_names
        this_mp.save_path = save_path
        jobs_to_do.append(this_mp)

    p = multiprocessing.Pool(multiprocessing.cpu_count()-2)
    p.map(mp_worker, jobs_to_do) # submit jobs_to_do
    p.close()