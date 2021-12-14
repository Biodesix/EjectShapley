# classification tree class
import crast.tree
import crast.node
import math
import numpy as np
import random
import itertools

class Tree(crast.tree.Tree):
    def __init__(self) -> None:
        super().__init__()
        # set up training parameters
        self.nclasses = -999
        self.definitions = [] # numeric definitions for faster entropy calc

        # hyperparameters
        self.max_depth = 1
        self.nfeatures_to_sample = -1
        self.min_leaf_size = 1

        # binning/encoding
        self.cont_ftr_bins = {} # dict from ftr name to bin array
        self.cat_ftr_encodings = {} # dict from ftr name to numeric encoding
        self.cat_ftr_groupings = {} # dict from ftr name to list of lists of possible categorical data groupings

    # read and prepare data methods
    def read_from_sampling_engine(self, in_se):
        self.data = in_se['data']
        self.ftr_names = in_se['ftr_names']
        self.is_ftr_categorical = in_se['is_ftr_categorical']
        def_map = {}
        for dd in self.data:
            if not dd['class'] in def_map:
                def_map[dd['class']] = len(def_map)
            self.definitions.append(def_map[dd['class']])
        self.nclasses = len(def_map)

    # training methods
    def fit(self) -> None:
        if self.nfeatures_to_sample < 0:
            self.nfeatures_to_sample = math.floor(math.sqrt(len(self.ftr_names)))

        this_node = crast.node.CrastNode()
        this_node.sample_idxs = [ii for ii in range(len(self.data))]
        this_node.depth = 0
        this_node.entropy = self.get_class_entropy(this_node.sample_idxs)
        this_node.id = 0
        this_node.sample_weight = len(this_node.sample_idxs)
        this_node.value = 0 # TODO: uninformative SV, might want to move this
        self.nodes.append(this_node)
        self.split_node(this_node.id)

    def split_node(self, parent_node_id) -> None:
        if self.nodes[parent_node_id].depth >= self.max_depth:
            return

        ftr_use_idxs = random.sample(range(len(self.ftr_names)), self.nfeatures_to_sample)

        optimal_ftr_idx = -1
        optimal_ftr_cutoff = -999
        optimal_split = {}

        max_info_gain = 0

        for iftr in ftr_use_idxs:
            if self.is_ftr_categorical[iftr]:
                pass
            else:
                for cutoff in self.cont_ftr_bins[self.ftr_names[iftr]]:
                    this_split = self.split_data(self.nodes[parent_node_id].sample_idxs, iftr, cutoff)
                    if len(this_split['left']) < self.min_leaf_size or len(this_split['right']) < self.min_leaf_size:
                        continue
                    this_ig = self.get_information_gain(this_split['left'], this_split['right'], parent_node_id)
                    if this_ig > max_info_gain:
                        optimal_ftr_idx = iftr
                        optimal_ftr_cutoff = cutoff
                        optimal_split = this_split
                        max_info_gain = this_ig
        
        if optimal_ftr_idx >= 0:

            self.nodes[parent_node_id].feature = optimal_ftr_idx
            self.nodes[parent_node_id].threshold = optimal_ftr_cutoff

            new_node_left = crast.node.CrastNode()
            new_node_left.sample_idxs = optimal_split['left']
            new_node_left.depth = self.nodes[parent_node_id].depth + 1
            new_node_left.entropy = self.get_class_entropy(optimal_split['left'])
            new_node_left.id = len(self.nodes)
            new_node_left.value = self.get_class_plurality(optimal_split['left'])
            new_node_left.sample_weight = len(optimal_split['left'])
            self.nodes[parent_node_id].child_left = new_node_left.id
            self.nodes.append(new_node_left)

            new_node_right = crast.node.CrastNode()
            new_node_right.sample_idxs = optimal_split['right']
            new_node_right.depth = self.nodes[parent_node_id].depth + 1
            new_node_right.entropy = self.get_class_entropy(optimal_split['right'])
            new_node_right.id = len(self.nodes)
            new_node_right.value = self.get_class_plurality(optimal_split['right'])
            self.nodes[parent_node_id].child_right = new_node_right.id
            new_node_right.sample_weight = len(optimal_split['right'])
            self.nodes.append(new_node_right)

            self.split_node(new_node_left.id)
            self.split_node(new_node_right.id)

    
    def split_data(self, sample_idxs, ftr_idx, cutoff):
        left_idxs = []
        right_idxs = []
        for idx in sample_idxs:
            if self.data[idx]['features'][ftr_idx] < cutoff:
                left_idxs.append(idx)
            else:
                right_idxs.append(idx)
        out = {}
        out['left'] = left_idxs
        out['right'] = right_idxs
        return out
    
    def split_data_categorical(self, sample_idxs, ftr_idx, grouping):
        left_idxs = []
        right_idxs = []
        for idx in sample_idxs:
            if self.data[idx]['features'][ftr_idx] in grouping:
                left_idxs.append(idx)
            else:
                right_idxs.append(idx)
        out = {}
        out['left'] = left_idxs
        out['right'] = right_idxs
        return out

    # feature binning and categorical data encoding
    def bin_continuous_features(self, nbins) -> None:
        for iftr, ftr_name in enumerate(self.ftr_names):
            if self.is_ftr_categorical[iftr]:
                continue
            ftr_list = []
            for dd in self.data:
                ftr_list.append(dd['features'][iftr])
            self.cont_ftr_bins[ftr_name] = [] 
            for ibin in range(nbins-1):
                self.cont_ftr_bins[ftr_name].append(np.percentile(np.array(ftr_list), 100*(ibin+1)/nbins))

    def encode_categorical_attributes(self) -> None:
        for iftr, ftr_name in enumerate(self.ftr_names):
            if not self.is_ftr_categorical[iftr]:
                continue
            cat_dict = {}
            for dd in self.data:
                if not dd['features'][iftr] in cat_dict:
                    cat_dict[dd['features'][iftr]] = len(cat_dict)
            self.cat_ftr_encodings[ftr_name] = cat_dict
    
    def calc_categorical_groupings(self) -> None:
        for iftr, ftr_name in enumerate(self.ftr_names):
            if not self.is_ftr_categorical[iftr]:
                continue
            ncats = len(self.cat_ftr_encodings[ftr_name])
            self.cat_ftr_groupings[ftr_name] = []
            # if ncats is even, only need the first half (icat from 1, ..., ncats/2 inclusive), then only half of ncats/2
            # if ncats is odd, only need less (icat from 1, ..., (ncats-1)/2 inclusive), then no funny business
            if ncats%2:
                for icat in range(1,(ncats+1)/2):
                    for combo in itertools.combinations(range(ncats), icat):
                        self.cat_ftr_groupings[ftr_name].append(list(combo))
            else:
                for icat in range(1,ncats/2+1):
                    ncombos = math.comb(ncats, icat)
                    for icombo, combo in enumerate(itertools.combinations(range(ncats), icat)):
                        if icat == ncats/2 and icombo >= ncombos:
                            break
                        self.cat_ftr_groupings[ftr_name].append(list(combo))

    # classification metric specific methods
    def get_class_entropy(self, idxs):
        counts = [0 for ii in range(self.nclasses)]
        for idx in idxs:
            counts[self.definitions[idx]] += 1
        out_entropy = 0
        for count in counts:
            this_prop = count/len(idxs)
            if this_prop > 0:
                out_entropy -= this_prop*math.log(this_prop)
        return out_entropy
    
    def get_information_gain(self, gr1_idxs, gr2_idxs, parent_node_id):
        s1 = self.get_class_entropy(gr1_idxs)
        s2 = self.get_class_entropy(gr2_idxs)
        n1 = len(gr1_idxs)
        n2 = len(gr2_idxs)
        s_ave = n1/(n1+n2)*s1 + n2/(n1+n2)*s2
        s_out = self.nodes[parent_node_id].entropy - s_ave
        return s_out
    
    def get_class_plurality(self, sample_idxs):
        counts = {}
        for idx in sample_idxs:
            this_def = self.data[idx]['class']
            if this_def in counts:
                counts[this_def] += 1
            else:
                counts[this_def] = 1
        plur_classes = []
        max_count = -1
        for cc in counts:
            if counts[cc] >= max_count:
                plur_classes.append(cc)
        plur_class = random.sample(plur_classes, 1)
        return plur_class[0]
    
    def use_binary_leaf_score(self, target_class):
        # overwrites class value in a binary problem with the proportion of samples with class target_class
        if len(self.nodes) == 0:
            raise Exception('this should only be used after a tree is grown')
        for inode, nn in enumerate(self.nodes):
            npos = 0
            for idx in nn.sample_idxs:
                if self.data[idx]['class'] == target_class:
                    npos += 1
            self.nodes[inode].value = npos/len(nn.sample_idxs)

    def get_class_prop(self, idxs, target_class):
        npos = 0
        for idx in idxs:
            if self.data[idx]['class'] == target_class:
                npos += 1
        return npos/len(idxs)