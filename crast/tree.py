# base tree class
import crast.node
import csv
import numpy as np
import random
import statistics

class Tree:
    def __init__(self) -> None:
        self.nodes = []

    # constructor from shap tree
    @classmethod
    def from_shap_tree(cls, in_tree):
        out = cls()
        # import shap
        for inode in range(len(in_tree.features)):
            this_node = crast.node.CrastNode()
            this_node.id = inode
            this_node.feature = in_tree.features[inode]
            this_node.threshold = in_tree.thresholds[inode]
            this_node.child_left = in_tree.children_left[inode]
            this_node.child_right = in_tree.children_right[inode]
            this_node.value = float(in_tree.values[inode])
            this_node.sample_weight = in_tree.node_sample_weight[inode]
            out.nodes.append(this_node)
        return out

    # constructor from sklearn shap tree
    @classmethod
    def from_sk_shap_tree(cls, in_tree):
        out = cls()
        # import shap
        for inode in range(len(in_tree.features)):
            this_node = crast.node.CrastNode()
            this_node.id = inode
            this_node.feature = in_tree.features[inode]
            this_node.threshold = in_tree.thresholds[inode]
            this_node.child_left = in_tree.children_left[inode]
            this_node.child_right = in_tree.children_right[inode]
            this_node.value = float(in_tree.values[inode][0])
            this_node.sample_weight = in_tree.node_sample_weight[inode]
            out.nodes.append(this_node)
        return out

    def print_tree(self) -> None:
        print('Printing Tree Info')
        print('-----------------------')
        for nn in self.nodes:
            print('Node: %i'%(nn.id))
            print('Feature: %i'%(nn.feature))
            print('Threshold: %f'%(nn.threshold))
            print('Child Left: %i'%(nn.child_left))
            print('Child Right: %i'%(nn.child_right))
            print('Value: %i'%(nn.value))
            print('-----------------------')

    def read_tree(self, in_path):
        header_idxs = {'child_left':-1, 'child_right':-1, 'feature':-1, 'threshold':-1, 'value':-1, 'sample_weight':-1, 'depth':-1}
        with open(in_path, 'r', newline='') as ifh:
            reader = csv.reader(ifh, delimiter=',', quotechar='|')
            for irow, row in enumerate(reader):
                if irow == 0:
                    for hidx in header_idxs:
                        for ii in range(len(row)):
                            if hidx == row[ii]:
                                header_idxs[hidx] = ii
                                break
                        if header_idxs[hidx] < 0:
                            print('could not find %s'%(hidx))
                            exit()
                else:
                    this_node = crast.node.CrastNode()
                    for hh in header_idxs:
                        this_node.set_value(hh, row[header_idxs[hh]])
                    self.nodes.append(this_node)

    def predict_tree(self, X):
        next_idx = 0
        while next_idx >= 0:
            this_idx = next_idx
            next_idx = self.nodes[next_idx].predict(X)
        return self.nodes[this_idx].value

    # early ejection prediction given a set of non-missing features indexed by ftr_idxs
    def predict_tree_eject(self, X, ftr_idxs):
        next_idx = 0
        while next_idx >= 0:
            this_idx = next_idx
            if not self.nodes[next_idx].feature in ftr_idxs:
                return self.nodes[next_idx].value
            next_idx = self.nodes[next_idx].predict(X)
        return self.nodes[this_idx].value

    # treeshap alg1 recursive method
    def ts_alg1_recursive(self, X, ftr_idxs, start_idx):
        if self.nodes[start_idx].feature < 0:
            return self.nodes[start_idx].value
        if self.nodes[start_idx].feature in ftr_idxs:
            next_idx = self.nodes[start_idx].predict(X)
            return self.ts_alg1_recursive(X, ftr_idxs, next_idx)
        else:
            left_idx = self.nodes[start_idx].child_left
            left_wt = len(self.nodes[left_idx].sample_idxs)/len(self.nodes[start_idx].sample_idxs)
            right_idx = self.nodes[start_idx].child_right
            right_wt = len(self.nodes[right_idx].sample_idxs)/len(self.nodes[start_idx].sample_idxs)
            return (left_wt*self.ts_alg1_recursive(X, ftr_idxs, left_idx) + right_wt*self.ts_alg1_recursive(X, ftr_idxs, right_idx))

    # treeshap alg1 prediction given a set of non-missing features indexed by ftr_idxs
    def predict_tree_ts_alg1(self, X, ftr_idxs):
        return self.ts_alg1_recursive(X, ftr_idxs, 0)

    # calc covariance matrix and anything else needed
    def init_covar_imputer(self, data) -> None:
        nsamples = len(data)
        nftrs = len(data[0]['features'])
        ftr_means = [0.0 for ii in range(nftrs)]
        for isample in range(nsamples):
            for iftr in range(nftrs):
                ftr_means[iftr] += data[isample]['features'][iftr]
        for iftr in range(nftrs):
            ftr_means[iftr] = ftr_means[iftr]/nsamples

        self.cov_mat = [[0.0 for jj in range(nftrs)] for ii in range(nftrs)]
        for isample in range(nsamples):
            for iftr in range(nftrs):
                for jftr in range(nftrs):
                    self.cov_mat[iftr][jftr] += (data[isample]['features'][iftr] - ftr_means[iftr])*(data[isample]['features'][jftr] - ftr_means[jftr])/(ftr_means[iftr]*ftr_means[jftr])

        for iftr in range(nftrs):
            for jftr in range(nftrs):
                self.cov_mat[iftr][jftr] = self.cov_mat[iftr][jftr]/nsamples
        self.chol = np.linalg.cholesky(np.array(self.cov_mat))
        self.ftr_lists = [[] for ii in range(nftrs)]
        for isample in range(nsamples):
            for iftr in range(nftrs):
                self.ftr_lists[iftr].append(data[isample]['features'][iftr])
        for iftr in range(nftrs):
            self.ftr_lists[iftr] = np.sort(self.ftr_lists[iftr])
        self.ftr_means = [0.0 for ii in range(nftrs)]
        self.ftr_stds = [0.0 for ii in range(nftrs)]
        for iftr in range(nftrs):
            self.ftr_means[iftr] = statistics.mean(self.ftr_lists[iftr])
            self.ftr_stds[iftr] = statistics.stdev(self.ftr_lists[iftr])

    def get_sig_val_from_feature(self, ftr_idx, value):
        return (value-self.ftr_means[ftr_idx])/self.ftr_stds[ftr_idx]

    def get_feature_from_sig_val(self, ftr_idx, value):
        return self.ftr_means[ftr_idx] + value*self.ftr_stds[ftr_idx]

    def predict_tree_covar_impute(self, X, ftr_idxs, nthrows):
        if len(ftr_idxs) == 0:
            return 0
        nftrs = len(X)
        nsamples = len(self.ftr_lists[0])
        throw_vec = [0.0 for ii in range(nftrs)]
        for idx in ftr_idxs:
            throw_vec[idx] = self.get_sig_val_from_feature(idx, X[idx])
        throw_idxs = []
        for iftr in range(nftrs):
            if not iftr in ftr_idxs:
                throw_idxs.append(iftr)
        pred_res = 0.0
        for ithrow in range(nthrows):
            ftr_res = [0.0 for ii in range(nftrs)]
            for idx in throw_idxs:
                # throw_vec[idx] = self.ftr_lists[idx][random.randint(0,nsamples-1)]
                throw_vec[idx] = random.gauss(0,1)
            for iftr in range(nftrs):
                for jftr in range(nftrs):
                    ftr_res[iftr] += throw_vec[jftr]*self.chol[iftr][jftr]
            for iftr in range(nftrs):
                ftr_res[iftr] = self.get_feature_from_sig_val(iftr, ftr_res[iftr])
            for iftr in ftr_idxs:
                ftr_res[iftr] = X[iftr]
            pred_res += self.predict_tree(ftr_res)
        return pred_res/nthrows

    def predict_tree_covar_impute_old(self, X, ftr_idxs, nthrows):
        if len(ftr_idxs) == 0:
            return 0
        nftrs = len(X)
        nsamples = len(self.ftr_lists[0])
        throw_vec = [0.0 for ii in range(nftrs)]
        for idx in ftr_idxs:
            throw_vec[idx] = self.get_sig_val_from_feature(idx, X[idx])
        throw_idxs = []
        for iftr in range(nftrs):
            if not iftr in ftr_idxs:
                throw_idxs.append(iftr)
        ftr_res = [0.0 for ii in range(nftrs)]
        for ithrow in range(nthrows):
            for idx in throw_idxs:
                # throw_vec[idx] = self.ftr_lists[idx][random.randint(0,nsamples-1)]
                throw_vec[idx] = random.gauss(0,1)
            for iftr in range(nftrs):
                for jftr in range(nftrs):
                    ftr_res[iftr] += throw_vec[jftr]*self.chol[iftr][jftr]
        for iftr in range(nftrs):
            ftr_res[iftr] = self.get_feature_from_sig_val(iftr, ftr_res[iftr]/nthrows)
        for iftr in ftr_idxs:
            ftr_res[iftr] = X[iftr]
        
        # maybe actually want average of predictions over throws rather than prediction of averaged throws
        return self.predict_tree(ftr_res)

    def init_ts_intervent(self, data):
        nsamples = len(data)
        nftrs = len(data[0]['features'])
        self.ref_data = [[0.0 for jj in range(nftrs)] for ii in range(nsamples)]
        for isample in range(nsamples):
            for iftr in range(nftrs):
                self.ref_data[isample][iftr] = data[isample]['features'][iftr]

    def predict_tree_ts_intervent(self, X, ftr_idxs):
        # expectation value of predicting with tree with missing features replaced by those in the training set
        fill_idxs = []
        nsamples = len(self.ref_data)
        nftrs = len(self.ref_data[0])
        for iftr in range(nftrs):
            if not iftr in ftr_idxs:
                fill_idxs.append(iftr)

        this_X = [X[ii] for ii in range(nftrs)]
        ave_pred = 0.0
        for isample in range(nsamples):
            for iftr in fill_idxs:
                this_X[iftr] = self.ref_data[isample][iftr]
            ave_pred += self.predict_tree(this_X)
        return ave_pred/nsamples