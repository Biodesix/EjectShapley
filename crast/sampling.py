import csv
import random
import math

class Engine:
    def __init__(self) -> None:
        self.is_ftr_categorical = []
        self.ftr_names = []
        self.data = []

        # sampling 
        self.train_idxs = []


    def read_data(self, in_path, ftr_names, id_name, **kwargs) -> None:

        header_names = {'id':'', 'features': [], 'class':'', 'val':'', 'surv_time':'', 'surv_cens':''}
        if type(ftr_names) == list:
            for ff in ftr_names:
                if not type(ff) == str:
                    raise Exception('ftr_names must be list of strs')
                else:
                    header_names['features'].append(ff)
        else:
            raise Exception('ftr_names must be list of strs')
        self.ftr_names = ftr_names

        if not type(in_path) == str:
            raise Exception('in_path must be str')

        if not type(id_name) == str:
            raise Exception('id_name must be str')
        header_names['id'] = id_name

        for key, value in kwargs.items():
            if key == 'ftr_is_categorical':
                if not type(value) == list:
                    raise Exception('ftr_is_categorical must be list of bools')
                if not type(value[0]) == bool:
                    raise Exception('ftr_is_categorical must be list of bools')
                if not len(value) == len(ftr_names):
                    raise Exception('ftr_is_categorical is inconsistent with number of features')
                self.is_ftr_categorical = value
            elif key == 'class_name':
                if not type(value) == str:
                    raise Exception('class_name must be str')
                header_names['class'] = value
            elif key == 'val_name':
                if not type(value) == str:
                    raise Exception('val_name must be str')
                header_names['val'] = value
            elif key == 'survival_name':
                if not type(value) == str:
                    raise Exception('survival_name must be str')
                header_names['surv_time'] = value
            elif key == 'survival_censor_name':
                if not type(value) == str:
                    raise Exception('survival_censor_name must be str')
                header_names['surv_cens'] = value
            else:
                raise Exception('invalid argument')

        if len(self.is_ftr_categorical) == 0:
            self.is_ftr_categorical = [False for ii in range(len(ftr_names))]

        with open(in_path, 'r', newline='') as ifh:
            reader = csv.reader(ifh, delimiter=',', quotechar='|')
            header_idxs = {'id':-1, 'features': [], 'class': -1, 'val':-1, 'surv_time':-1, 'surv_cens':-1}
            for irow, row in enumerate(reader):
                if irow == 0:
                    for hh in header_idxs:
                        if hh == 'features':
                            for ftr_name in header_names[hh]:
                                for ir, rr in enumerate(row):
                                    if ftr_name == rr:
                                        header_idxs['features'].append(ir)
                                        break
                        else:
                            for ir, rr in enumerate(row):
                                if header_names[hh] == rr:
                                    header_idxs[hh] = ir
                                    break
                else:
                    this_sample = {'id':'NA', 'features': [], 'class': 'NA', 'val':-999, 'surv_time':-999, 'surv_cens':-999}
                    for key, value in header_idxs.items():
                        if key == 'features':
                            for ii, idx in enumerate(value):
                                if self.is_ftr_categorical[ii]:
                                    this_sample['features'].append(str(row[idx]))
                                else:
                                    this_sample['features'].append(float(row[idx]))
                        elif key == 'id' or key == 'class':
                            if value >= 0:
                                this_sample[key] = str(row[value])
                        else:
                            if value >= 0:
                                this_sample[key] = float(row[value])

                    self.data.append(this_sample)
    

    def set_seed(self, seed):
        random.seed(seed)

    def generate_sampling(self, train_fraction) -> None:
        # generate sampling
        gr_idxs = {}
        for ii, dd in enumerate(self.data):
            if dd['class'] in gr_idxs:
                gr_idxs[dd['class']].append(ii)
            else:
                gr_idxs[dd['class']] = []
                gr_idxs[dd['class']].append(ii)
        min_gr_count = 1e6
        for gg in gr_idxs:
            if len(gr_idxs[gg]) < min_gr_count:
                min_gr_count = len(gr_idxs[gg])
        n_to_sample = math.floor(train_fraction*min_gr_count)
        self.train_idxs = []
        for gg in gr_idxs:
            g_idxs = random.sample(gr_idxs[gg], n_to_sample)
            for idx in g_idxs:
                self.train_idxs.append(idx)

    def get_training_samples(self):
        # return training samples for tree along with other things it wants
        out = {}
        out['data'] = [self.data[ii] for ii in self.train_idxs]
        out['ftr_names'] = self.ftr_names
        out['is_ftr_categorical'] = self.is_ftr_categorical
        return out

    # methods for out-of-bag performance estimates
    def get_oob_samples(self):
        # return the oob samples
        pass

    def set_oob_result(self, in_result):
        # set oob predictions
        pass