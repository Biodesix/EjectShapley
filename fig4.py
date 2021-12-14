import os
import random
import pickle
import multiprocessing
import crast.sampling
import crast.ctree
import crast.shap

def calc_SVs(ftr_names, nbags, train_path, test_path, save_path, job_name):
    se = crast.sampling.Engine()
    se.read_data(train_path, ftr_names, 'Filename', class_name='2yr_rec')

    se_val = crast.sampling.Engine()
    se_val.read_data(test_path, ftr_names, 'Filename', class_name='2yr_rec')
    test_set = []
    for ss in se_val.data:
        test_set.append(ss['features'])

    nsamples = len(test_set)
    nfeatures = len(ftr_names)


    for ibag in range(nbags):
        out_res = {}
        out_res['SVs_ts'] = [[-999.9 for jj in range(nfeatures)] for ii in range(nsamples)]
        out_res['SVs_ej'] = [[-999.9 for jj in range(nfeatures)] for ii in range(nsamples)]
        out_res['probs'] = [0.0 for ii in range(nsamples)]
        out_res['used_ftrs'] = [[] for ii in range(nsamples)]

        se.generate_sampling(0.667)

        ct = crast.ctree.Tree()
        ct.read_from_sampling_engine(se.get_training_samples())
        ct.bin_continuous_features(10)
        ct.fit()

        st = crast.shap.Tree.from_tree(ct)
        st.shap_init()

        bias = 0
        for it, tt in enumerate(test_set):
            svs_ts = st.shap_values(tt)
            out_res['probs'][it] = st.predict_tree(tt)
            # get bias and correct it
            if it == 0:
                if sum(svs_ts) < 0:
                    bias = sum(svs_ts) + 1
                else:
                    bias = sum(svs_ts) - 1

            for ii in range(len(svs_ts)):
                svs_ts[ii] -= bias/len(svs_ts)
            
            svs_ej = st.shap_values_eject_path_new(tt)
            out_res['used_ftrs'][it] = st.get_ftr_path(tt)

            for iftr in range(len(ftr_names)):
                out_res['SVs_ts'][it][iftr] = svs_ts[iftr]
                out_res['SVs_ej'][it][iftr] = svs_ej[iftr]

        with open('%s\\SVs_%s_%i.p'%(save_path, job_name, ibag), 'wb') as ofile:
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

# collect and combine results
def combine_results(in_dir, out_fname, test_path):
    se_val = crast.sampling.Engine()

    se_val.read_data(test_path, ['ftr_1'], 'Filename', class_name='2yr_rec')

    nfiles = 0
    ifile = 0
    for file in os.listdir(in_dir):
        if 'combined' in file:
            continue
        if ifile%10 == 0:
            print('on file: %i'%(ifile))
        SVs_path = in_dir + '/' + file
        with open(SVs_path, 'rb') as infile:
            nfiles += 1
            this_res = pickle.load(infile)
            if ifile == 0:
                nsamples = len(this_res['SVs_ts'])
                nftrs = len(this_res['SVs_ts'][0])
                SVs_ts = [[0.0 for ii in range(nftrs)] for jj in range(nsamples)]
                SVs_ej = [[0.0 for ii in range(nftrs)] for jj in range(nsamples)]
                probs = [0.0 for ii in range(nsamples)]
                used_ftrs = [[] for jj in range(nsamples)]
            for isample in range(nsamples):
                probs[isample] += this_res['probs'][isample]
                for idx in this_res['used_ftrs'][isample]:
                    used_ftrs[isample].append(idx)
                for iftr in range(nftrs):
                    SVs_ts[isample][iftr] += this_res['SVs_ts'][isample][iftr]
                    SVs_ej[isample][iftr] += this_res['SVs_ej'][isample][iftr]
        ifile += 1

    for isample in range(nsamples):
        probs[isample] = probs[isample]/nfiles
        used_ftrs[isample] = list(set(used_ftrs[isample]))
        for iftr in range(nftrs):
            SVs_ts[isample][iftr] = SVs_ts[isample][iftr]/nfiles
            SVs_ej[isample][iftr] = SVs_ej[isample][iftr]/nfiles

    test_set_defs = []

    for ss in se_val.data:
        test_set_defs.append(int(ss['class']))

    labels = [-999 for ii in range(nsamples)]
    for ii in range(nsamples):
        if probs[ii] == 0.0:
            temp = random.sample([0,1], 1)
            labels[ii] = temp[0]
        elif probs[ii] < 0:
            labels[ii] = 0
        else:
            labels[ii] = 1
    acc = 0
    for ii in range(nsamples):
        if labels[ii] == test_set_defs[ii]:
            acc += 1/nsamples
    print('accuracy was: %f'%(acc))

    out_res = {}
    out_res['probs'] = probs
    out_res['labels'] = labels
    out_res['SVs_ts'] = SVs_ts
    out_res['SVs_ej'] = SVs_ej
    out_res['defs'] = test_set_defs
    out_res['used_ftrs'] = used_ftrs
    with open(out_fname, 'wb') as ofile:
        pickle.dump(out_res, ofile)

# generate processing pool, run, and combine results
if __name__ == '__main__':
    print('Welcome to figure 4.  This produces an abbreviated set of data compared to what is in manuscript.  Please see source code for details')
    nbags_per = 10
    njobs = 10

    # what was run for paper, 10k trees over 100 jobs
    # nbags_per = 100
    # njobs = 100
    # as (naively) implemented, each tree written will contain about 1 MB of SVs (12770x380 matrix).  Obviously suboptimal, but useful to make
    # sure the algorithms are infact giving zeros where they should.
    # provided combined data file is complete and is what was used for the paper

    save_path = './results/fig4'
    train_path = './data/NKI_cleaned.csv'
    test_path = './data/LOI_cleaned.csv'

    ftr_names = []
    for ii in range(1, 12771):
        ftr_names.append('ftr_%i'%(ii))

    jobs_to_do = []
    for ii in range(njobs):
        this_mp = mp_worker_input(str(ii), ii)
        this_mp.nbags = nbags_per
        this_mp.train_path = train_path
        this_mp.test_path = test_path
        this_mp.ftr_names = ftr_names
        this_mp.save_path = save_path
        jobs_to_do.append(this_mp)

    p = multiprocessing.Pool(multiprocessing.cpu_count()-2) # all cores on machine - 2 
    p.map(mp_worker, jobs_to_do) # submit jobs_to_do
    p.close()

    print('collating results')
    out_fname = './results/fig4/NKI_dev_combined_100_trees.p'
    combine_results(save_path, out_fname, test_path) 