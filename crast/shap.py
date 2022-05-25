# tree class for calcuating shapley values
import crast.tree
import numpy as np
from crast.tree_shap import tree_shap_recursive
import itertools
import math
import pickle

class Tree(crast.tree.Tree):

    def __init__(self) -> None:
        super().__init__()
        self.prefactors = {}
    
    # constructor from other tree
    @classmethod
    def from_tree(cls, in_tree):
        out = cls()
        out.nodes = in_tree.nodes
        # convert 0,1 values (defs) to -1,1
        for ii, nn in enumerate(out.nodes):
            if ii == 0:
                out.nodes[ii].value = 0
            else:
                if float(nn.value) < 0.5:
                    out.nodes[ii].value = -1
                else:
                    out.nodes[ii].value = 1
        return out

    # constructor from other tree (scored)
    @classmethod
    def from_scored_tree(cls, in_tree):
        out = cls()
        out.nodes = in_tree.nodes
        # convert 0,1 values to -1,1
        for ii, nn in enumerate(out.nodes):
            out.nodes[ii].value = (nn.value-0.5)*2
        return out

    # constructor from regression tree
    @classmethod
    def from_regression_tree(cls, in_tree):
        out = cls()
        out.nodes = in_tree.nodes
        return out


    def load_prefactors_from_tree(self, in_tree) -> None:
        for key in in_tree.prefactors:
            if not key in self.prefactors:
                self.prefactors[key] = in_tree.prefactors[key]

    def load_prefactors_from_file(self, in_path) -> None:
        with open(in_path, 'rb') as in_file:
            st = pickle.load(in_file)
            for key in st.prefactors:
                if not key in self.prefactors:
                    self.prefactors[key] = st.prefactors[key]
        print(self.prefactors)

    def shap_init(self):
        maxd = np.max([nn.depth for nn in self.nodes]) + 2
        s = (maxd * (maxd + 1)) // 2
        self.feature_indexes = np.zeros(s, dtype=np.int32)
        self.zero_fractions = np.zeros(s, dtype=np.float64)
        self.one_fractions = np.zeros(s, dtype=np.float64)
        self.pweights = np.zeros(s, dtype=np.float64)

        self.children_left = np.array([nn.child_left for nn in self.nodes])
        self.children_right = np.array([nn.child_right for nn in self.nodes])
        self.children_default = np.array([nn.child_left for nn in self.nodes])
        self.features= np.array([nn.feature for nn in self.nodes])
        self.thresholds = np.array([nn.threshold for nn in self.nodes])
        self.values = np.array([nn.value for nn in self.nodes])
        self.node_sample_weight = np.array([nn.sample_weight for nn in self.nodes])

    def shap_values(self, X):
        self.shap_init()
        # only single instance
        # phi = np.zeros(len(X))
        phi = np.zeros([len(X), 1])
        x_missing = np.zeros(len(X), dtype=np.bool)
        condition = 0
        condition_feature = 0

        # update the bias term, which is the last index in phi
        # (note the paper has this as phi_0 instead of phi_M)
        # phi[-1,:] += self.predict_tree(X) 
        # phi[-1,:] = 0.5 # this doesn't seem to do anything....

        # start the recursive algorithm
        tree_shap_recursive(
            self.children_left, self.children_right, self.children_default, self.features,
            self.thresholds, self.values, self.node_sample_weight,
            X, x_missing, phi, 0, 0, self.feature_indexes, self.zero_fractions, self.one_fractions, self.pweights,
            1, 1, -1, condition, condition_feature, 1
        )

        return phi[:,0]

    # brute force summed SVs using eject predict
    def shap_values_eject_brute(self, X):
        svs = [0 for ii in range(len(X))]
        for iftr in range(len(X)):
            ftr_idxs = []
            for jftr in range(len(X)):
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(X)):
                pre_factor = 1.0/(len(X)*math.comb(len(X) - 1, isize))
                if isize == 0:
                    # zeroth order term
                    svs[iftr] += pre_factor*(self.predict_tree_eject(X, [iftr]) - self.predict_tree_eject(X, []))
                else:
                    for combo in itertools.combinations(ftr_idxs, isize):
                        plus_set = list(combo)
                        plus_set.append(iftr)
                        svs[iftr] += pre_factor*(self.predict_tree_eject(X, plus_set) - self.predict_tree_eject(X, list(combo)))
        return svs

    # brute force summed SVs using covar impute predict
    def shap_values_covar_impute_brute(self, X, nthrows):
        svs = [0 for ii in range(len(X))]
        for iftr in range(len(X)):
            ftr_idxs = []
            for jftr in range(len(X)):
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(X)):
                pre_factor = 1.0/(len(X)*math.comb(len(X) - 1, isize))
                if isize == 0:
                    # zeroth order term
                    svs[iftr] += pre_factor*(self.predict_tree_covar_impute(X, [iftr], nthrows) - self.predict_tree_covar_impute(X, [], nthrows))
                else:
                    for combo in itertools.combinations(ftr_idxs, isize):
                        plus_set = list(combo)
                        plus_set.append(iftr)
                        svs[iftr] += pre_factor*(self.predict_tree_covar_impute(X, plus_set, nthrows) - self.predict_tree_covar_impute(X, list(combo), nthrows))
        return svs

    # brute force summed SVs using ts interventional predict
    def shap_values_ts_intervent_brute(self, X):
        svs = [0 for ii in range(len(X))]
        for iftr in range(len(X)):
            ftr_idxs = []
            for jftr in range(len(X)):
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(X)):
                pre_factor = 1.0/(len(X)*math.comb(len(X) - 1, isize))
                if isize == 0:
                    # zeroth order term
                    svs[iftr] += pre_factor*(self.predict_tree_ts_intervent(X, [iftr]) - self.predict_tree_ts_intervent(X, []))
                else:
                    for combo in itertools.combinations(ftr_idxs, isize):
                        plus_set = list(combo)
                        plus_set.append(iftr)
                        svs[iftr] += pre_factor*(self.predict_tree_ts_intervent(X, plus_set) - self.predict_tree_ts_intervent(X, list(combo)))
        return svs

    # path summed SVs using eject predict, fast
    def shap_values_eject_path_fast(self, X):

        # get feature path
        use_idxs = []
        node_vals = []
        next_idx = 0
        last_idx = 0
        ftr_counts = {}
        while next_idx >= 0:
            this_ftr = self.nodes[last_idx].feature
            node_vals.append(self.nodes[last_idx].value)
            if this_ftr in ftr_counts:
                ftr_counts[this_ftr] += 1
            else:
                ftr_counts[this_ftr] = 1
            use_idxs.append(this_ftr)
            last_idx = next_idx
            next_idx = self.nodes[next_idx].predict(X)
        
        # get unique path that removes duplicates leaving last instance in the path
        unique_path = []
        unique_vals = []
        for ii, idx in enumerate(use_idxs):
            if ftr_counts[idx] > 1:
                ftr_counts[idx] = ftr_counts[idx] - 1
            else:
                unique_path.append(idx)
                unique_vals.append(node_vals[ii])

        svs = [0 for ii in range(len(X))]

        # multiply coefficients into vals
        for ii in range(len(unique_vals)):
            if ii > 0:
                unique_vals[ii] = unique_vals[ii]/(ii*(ii+1))

        for iftr, ftr_idx in enumerate(unique_path):
            svs[ftr_idx] = self.predict_tree(X)/len(unique_path) - unique_vals[iftr]
            for val in unique_vals[iftr+1:]:
                svs[ftr_idx] += val
        return svs


    # path summed SVs using eject predict
    def shap_values_eject_path(self, X):
        svs = [0 for ii in range(len(X))]

        # get path ftr set
        use_idxs = []
        next_idx = 0
        last_idx = 0
        while next_idx >= 0:
            use_idxs.append(self.nodes[last_idx].feature)
            last_idx = next_idx
            next_idx = self.nodes[next_idx].predict(X)

        use_idxs = list(set(use_idxs))

        for iftr in use_idxs:
            ftr_idxs = []
            for jftr in use_idxs:
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(use_idxs)):
                for combo in itertools.combinations(ftr_idxs, isize):
                    plus_set = list(combo)
                    plus_set.append(iftr)
                    svs[iftr] += self.get_prefactor(len(X), len(use_idxs), isize)/len(X)*(self.predict_tree_eject(X, plus_set) - self.predict_tree_eject(X, list(combo)))
        return svs


    # path summed SVs using eject predict
    def shap_values_eject_path_legacy(self, X):
        svs = [0 for ii in range(len(X))]

        # get path ftr set
        use_idxs = []
        next_idx = 0
        last_idx = 0
        while next_idx >= 0:
            use_idxs.append(self.nodes[last_idx].feature)
            last_idx = next_idx
            next_idx = self.nodes[next_idx].predict(X)

        use_idxs = list(set(use_idxs))

        for iftr in use_idxs:
            ftr_idxs = []
            for jftr in use_idxs:
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(use_idxs)):
                for combo in itertools.combinations(ftr_idxs, isize):
                    extra_factor = 1.0/(len(X)*math.comb(len(X) - 1, isize))
                    for jsize in range(1, len(X)-isize):
                        pre_factor = 1.0/(len(X)*math.comb(len(X) - 1, isize+jsize))
                        extra_factor += pre_factor*math.comb((len(X)-len(use_idxs)), jsize)
                    plus_set = list(combo)
                    plus_set.append(iftr)
                    svs[iftr] += extra_factor*(self.predict_tree_eject(X, plus_set) - self.predict_tree_eject(X, list(combo)))
        return svs

    # path summed SVs using eject predict
    def shap_values_eject_path_new(self, X):
        svs = [0 for ii in range(len(X))]
        use_idxs = self.get_ftr_path(X)

        for iftr in use_idxs:
            ftr_idxs = []
            for jftr in use_idxs:
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(use_idxs)):
                pre_factor = 1.0/(len(use_idxs)*math.comb(len(use_idxs) - 1, isize))
                for combo in itertools.combinations(ftr_idxs, isize):
                    plus_set = list(combo)
                    plus_set.append(iftr)
                    svs[iftr] += pre_factor*(self.predict_tree_eject(X, plus_set) - self.predict_tree_eject(X, list(combo)))
        return svs

    # path summed SVs using eject predict
    def get_ftr_path(self, X):
        # get path ftr set
        use_idxs = []
        next_idx = 0
        last_idx = 0
        while next_idx >= 0:
            use_idxs.append(self.nodes[last_idx].feature)
            last_idx = next_idx
            next_idx = self.nodes[next_idx].predict(X)

        use_idxs = list(set(use_idxs))
        return use_idxs

    def float_comb(self, n, k):
        out = 1.0
        for ii in range(k):
            out *= (n-ii)/(ii+1)
        return out
    
    def compute_prefactor_new(self, nftrs, path_len, set_len):
        print('computing prefactor: %i, %i'%(path_len, set_len))
        out = 0
        prefactor = 1.0/self.float_comb(nftrs-1,set_len)
        out += prefactor
        for jsize in range(nftrs-set_len):
            prefactor *= (nftrs-path_len-jsize)*(set_len+1+jsize)/((jsize+1)*(nftrs-1-set_len-jsize))
            out += prefactor
        self.prefactors['%i%i'%(path_len, set_len)] = out
   
    def compute_prefactor(self, nftrs, path_len, set_len):
        print('computing prefactor: %i, %i'%(path_len, set_len))
        out = 0
        prefactor = 1.0/self.float_comb(nftrs-1,set_len)
        out += prefactor
        for jsize in range(1,nftrs-set_len):
            out += self.float_comb(nftrs-path_len, jsize)/self.float_comb(nftrs-1,set_len+jsize)
        self.prefactors['%i%i'%(path_len, set_len)] = out

    def compute_prefactors(self, nftrs, max_path_len):
        # computes all prefactors for feature paths up to length max_path_len given nftrs
        for ilen in range(1, max_path_len+1):
            for jlen in range(0, ilen):
                self.compute_prefactor(nftrs, ilen, jlen)

    def get_prefactor(self, nftrs, path_len, set_len):
        key = '%i%i'%(path_len, set_len)
        if key in self.prefactors:
            return self.prefactors[key]
        else:
            self.compute_prefactor(nftrs, path_len, set_len)
            return self.prefactors[key]

    # brute force summed SVs using treeshap alg1 predict
    def shap_values_ts_alg1_brute(self, X):
        svs = [0 for ii in range(len(X))]
        for iftr in range(len(X)):
            ftr_idxs = []
            for jftr in range(len(X)):
                if not jftr == iftr:
                    ftr_idxs.append(jftr)
            for isize in range(len(X)):
                pre_factor = 1.0/(len(X)*math.comb(len(X) - 1, isize))
                if isize == 0:
                    # zeroth order term
                    svs[iftr] += pre_factor*(self.predict_tree_ts_alg1(X, [iftr]) - self.predict_tree_ts_alg1(X, []))
                else:
                    for combo in itertools.combinations(ftr_idxs, isize):
                        plus_set = list(combo)
                        plus_set.append(iftr)
                        svs[iftr] += pre_factor*(self.predict_tree_ts_alg1(X, plus_set) - self.predict_tree_ts_alg1(X, list(combo)))
        return svs