import logging
import numpy as np
from joblib import Parallel, delayed
import torch
from torch_geometric.data import Data, DataLoader
from lookatthisgraph.utils.datautils import get_input_data, time_to_position, get_dom_xyz, get_edge_information, filter_pulses, build_data_list
from lookatthisgraph.utils.pulsenormalizer import PulseNormalizer


class Dataset(object):

    """Dataset for training of neural net"""

    def __init__(self, config):
        self.files = config['file_list']
        self.input_labels = ['energy', 'zenith']
        self.training_labels = config['training_labels']
        if 'n_jobs' in config:
            self.n_jobs = config['n_jobs']
        else:
            self.n_jobs = 1
        if 'max_pulses' in config:
            self.max_pulses = config['max_pulses']
        else:
            self.max_pulses = float('inf')

        self.include_charge = config['include_charge']

        raw_pulses, raw_truths = get_input_data(self.files, self.input_labels, self.n_jobs)
        self.filtered_pulses, self.filtered_truths, self.filter_mask = \
                filter_pulses(raw_pulses, raw_truths, self.n_jobs)

        self._n_events = len(self.filtered_pulses)
        logging.info('%i events' % self._n_events)

        transformed_truths = self._transform_truths()

        self.dom_xyz = get_dom_xyz(self.filtered_pulses, 0)
        normalized_features = self._get_normalized_features()

        self._data_list = build_data_list(
            normalized_features,
            transformed_truths
        )

        self.reshuffle()


    def reshuffle(self):
        self._permutation = np.random.permutation(self._n_events)


    def get_loaders(self, batch_size, train_split, test_split, val_split='batch'):
        split = lambda s: int(self._n_events * s) if s < 1 else s

        if val_split == 'batch':
            n_val = batch_size
        else:
            n_val = split(val_split)
        n_train, n_test = split(train_split), split(test_split)
        if n_train + n_val + n_test > self._n_events:
            raise ValueError('Loader configuration exceeds number of data samples')
        dataset_shuffled = [self._data_list[i] for i in self._permutation]

        train_loader = DataLoader(dataset_shuffled[:n_train], batch_size=batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(dataset_shuffled[n_train:n_train+n_val], batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(dataset_shuffled[n_train+n_val:n_train], batch_size=batch_size, drop_last=True)

        return train_loader, val_loader, test_loader


    def _get_normalized_features(self):
        if self.include_charge:
            features = [np.concatenate((np.log10(event[:, 2].reshape(-1, 1)), # log charges
                                        event[:, 1].reshape(-1, 1), # Time
                                        doms, # xyz
                                       ), axis=1)
                        for event, doms in zip(self.filtered_pulses, self.dom_xyz)]
        else:
            features = [np.concatenate((event[:, 1].reshape(-1, 1), # Time
                                        doms, # xyz
                                       ), axis=1)
                        for event, doms in zip(self.filtered_pulses, self.dom_xyz)]

        pn = PulseNormalizer(features)
        features_normalized = pn.normalize(mode='gauss')
        return features_normalized


    def _transform_truths(self):
        if self.training_labels == ['zenith']:
            col = np.where(self.input_labels == 'zenith')[0]
            transformed_truths = [[np.sin(y[col]), np.cos(y[col])] for y in self.filtered_truths]

        elif self.training_labels == ['energy']:
            col = np.where(self.input_labels == 'energy')[0]
            transformed_truths = [np.log10(y[col]) for y in self.filtered_truths]

        return transformed_truths
