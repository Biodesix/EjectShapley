#crast classes

#fig 2 and 4
train_path = '../data/NKI_cleaned.csv'
test_path = '../data/LOI_cleaned.csv'
# now in ./data/
#fig 3
SVs_path = '../results/fig1_30_new_1000'
# now in ./results/fig3
fnames = ['0p25', '0p50', '0p75', '1p0'] 
for ii, name in enumerate(fnames):
    train_path = '\\\\biodnas9\\ActiveProjects\\ShapleyValues\\TreeShap\\Experiments\\parallel_SVs\\data\\fig1_30_new\\train_%s.csv'%(name)
    test_path = '\\\\biodnas9\\ActiveProjects\\ShapleyValues\\TreeShap\\Experiments\\parallel_SVs\\data\\fig1_30_new\\test_%s.csv'%(name)
# now in ./data/fig3

#fig 4
path = '../results/NKI_dev_combined_full.p'
# now in ./results

#fig 5 and E2
train_path = '../data/NHANESI_training.csv'
test_path = '../data/NHANESI_validation.csv'
# now in ./data

#fig E1
in_dir = '../results/fig2_intervent/uncorr'
in_dir = '../results/fig2_intervent/corr'
# now in ./results/figE1/uncorr
# now in ./results/figE1/corr
train_path = '\\\\biodnas9\\ActiveProjects\\ShapleyValues\\TreeShap\\Experiments\\parallel_SVs\\data\\train_fig2.csv'
test_path = '\\\\biodnas9\\ActiveProjects\\ShapleyValues\\TreeShap\\Experiments\\parallel_SVs\\data\\test_fig2.csv'
# now in ./data/train_figE1_corr.csv
# now in ./data/train_figE1_uncorr.csv