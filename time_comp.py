import crast.sampling
import crast.ctree
import crast.shap
import time
from matplotlib import pyplot as plt
import statistics

se = crast.sampling.Engine()
se_val = crast.sampling.Engine()

# nbags = 100
# train_path = './data/NHANESI_training.csv'
# test_path = './data/NHANESI_validation.csv'
# ftr_names = ['sex_isFemale', 'age', 'physical_activity', 'alkaline_phosphatase', 'SGOT', 'BUN', 'calcium', 'creatinine', 'potassium', 'sodium', 'total_bilirubin', 'red_blood_cells', 'white_blood_cells', 'hemoglobin', 'hematocrit', 'segmented_neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils', 'band_neutrophils', 'cholesterol', 'urine_pH', 'uric_acid', 'systolic_blood_pressure', 'pulse_pressure', 'bmi']
# se.read_data(train_path, ftr_names, 'SampleID', class_name='death')
# se_val.read_data(test_path, ftr_names, 'SampleID', class_name='death')

# nbags = 100
# train_path = './data/NKI_cleaned.csv'
# test_path = './data/LOI_cleaned.csv'
# ftr_names = []
# for ii in range(1, 12770):
#     ftr_names.append('ftr_%i'%(ii))
# se.read_data(train_path, ftr_names, 'Filename', class_name='2yr_rec')
# se_val.read_data(test_path, ftr_names, 'Filename', class_name='2yr_rec')

nbags = 100
train_path = './data/train_figE1_uncorr.csv'
test_path = './data/test_figE1_uncorr.csv'
ftr_names = []
for ii in range(1, 14):
    ftr_names.append('IU%i'%(ii))
se.read_data(train_path, ftr_names, 'SampleID', class_name='Definition')
se_val.read_data(test_path, ftr_names, 'SampleID', class_name='Definition')

test_set = []
for ss in se_val.data:
    test_set.append(ss['features'])

times_ts = []
times_ej = []

for ibag in range(nbags):
    print(ibag)
    se.generate_sampling(0.667)

    ct = crast.ctree.Tree()
    ct.read_from_sampling_engine(se.get_training_samples())
    ct.bin_continuous_features(10)
    ct.fit()

    st = crast.shap.Tree.from_tree(ct)
    st.shap_init()
    for it, tt in enumerate(test_set):
        t0 = time.time()
        svs_ts = st.shap_values(tt)
        t1 = time.time()
        svs_ej = st.shap_values_eject_path_new(tt)
        t2 = time.time()

        times_ts.append(t1-t0)
        times_ej.append(t2-t1)

# plt.figure()
# plt.hist(times_ts, label='TreeShap: %e'%(statistics.mean(times_ts)), alpha=0.5) 
# plt.hist(times_ej, label='Eject: %e'%(statistics.mean(times_ej)), alpha=0.5)
# plt.xlabel('Time')
# plt.legend()
# plt.savefig('./plots/time_comp.png')
# plt.clf()
# plt.close()

print('TreeShap ave time: %e'%(statistics.mean(times_ts)))
print('Eject ave time: %e'%(statistics.mean(times_ej)))
print('TS/EJ: %.3f'%(statistics.mean(times_ts)/statistics.mean(times_ej)))

# import numpy as np
# diffs = []
# for ii in range(len(times_ts)):
#     diffs.append((times_ts[ii]-times_ej[ii])/times_ej[ii])

# print('Diff median (95%% range): %e (%e-%e)'%(np.median(diffs), np.percentile(diffs,2.5), np.percentile(diffs, 97.5)))