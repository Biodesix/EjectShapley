import pickle
import multiprocessing
import crast.sampling
import crast.ctree
import crast.shap

def calc_SVs(ftr_names, nbags, train_path, test_path, save_path, job_name):
    se = crast.sampling.Engine()
    se.read_data(train_path, ftr_names, 'SampleID', class_name='Definition')

    se_val = crast.sampling.Engine()
    se_val.read_data(test_path, ftr_names, 'SampleID', class_name='Definition')
    test_set = []
    for ss in se_val.data:
        test_set.append(ss['features'])
    nsamples = len(test_set)
    nfeatures = len(ftr_names)

    SVs_ts = [[0.0 for jj in range(nfeatures)] for ii in range(nsamples)]
    # SVs_ts_int = [[0.0 for jj in range(nfeatures)] for ii in range(nsamples)]
    SVs_ej = [[0.0 for jj in range(nfeatures)] for ii in range(nsamples)]

    for ibag in range(nbags):
        se.generate_sampling(0.667)

        ct = crast.ctree.Tree()
        ct.read_from_sampling_engine(se.get_training_samples())
        ct.bin_continuous_features(10)
        ct.fit()

        print('job: %s on bag %i'%(job_name, ibag))
        st = crast.shap.Tree.from_tree(ct)
        st.shap_init()
        st.init_ts_intervent(se.data)
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
            # svs_ts_int = st.shap_values_ts_intervent_brute(tt)

            for iftr in range(len(ftr_names)):
                SVs_ts[it][iftr] += svs_ts[iftr]/nbags
                # SVs_ts_int[it][iftr] += svs_ts_int[iftr]/nbags
                SVs_ej[it][iftr] += svs_ej[iftr]/nbags

    out_res = {}
    out_res['SVs_ts'] = SVs_ts
    # out_res['SVs_ts_int'] = SVs_ts_int
    out_res['SVs_ej'] = SVs_ej
    with open('%s/SVs_%s.p'%(save_path, job_name), 'wb') as ofile:
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


if __name__ == '__main__':
    print('Welcome to figure E1.  This produces an abbreviated set of data compared to what is in manuscript.  Please see source code for details')
    nbags = 10
    njobs = 10 # 100 used in manuscript
    # results in ./results/figE1/* are full and what was used in manuscript

    save_path = './results/figE1_test/uncorr'
    train_path = './data/train_figE1_uncorr.csv'
    test_path = './data/test_figE1_uncorr.csv'
    ftr_names = []
    for ii in range(1, 14):
        ftr_names.append('IU%i'%(ii))

    jobs_to_do = []
    for ii in range(njobs):
        this_mp = mp_worker_input(str(ii), ii)
        this_mp.nbags = nbags
        this_mp.train_path = train_path
        this_mp.test_path = test_path
        this_mp.ftr_names = ftr_names
        this_mp.save_path = save_path
        jobs_to_do.append(this_mp)
    p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    p.map(mp_worker, jobs_to_do) # submit jobs_to_do
    p.close()

    save_path = './results/figE1_test/corr'
    train_path = './data/train_figE1_corr.csv'
    test_path = './data/test_figE1_corr.csv'
    ftr_names = []
    for ii in range(1, 14):
        ftr_names.append('IC%i'%(ii))

    jobs_to_do = []
    for ii in range(njobs):
        this_mp = mp_worker_input(str(ii), ii)
        this_mp.nbags = nbags
        this_mp.train_path = train_path
        this_mp.test_path = test_path
        this_mp.ftr_names = ftr_names
        this_mp.save_path = save_path
        jobs_to_do.append(this_mp)
    p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    p.map(mp_worker, jobs_to_do) # submit jobs_to_do
    p.close()