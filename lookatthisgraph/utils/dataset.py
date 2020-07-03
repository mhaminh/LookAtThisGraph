import os
import logging
from importlib.resources import path
import numpy as np
# from joblib import Parallel, delayed
import torch
import lookatthisgraph.resources
from torch_geometric.data import Data, DataLoader
from lookatthisgraph.utils.datautils import process_charges, build_data_list
from lookatthisgraph.utils.pulsenormalizer import PulseNormalizer
from lookatthisgraph.utils.i3cols_dataloader import load_events


class Dataset(object):

    """Dataset for training of neural net"""

    def __init__(self, file_list, normalization_parameters=None):
        self.files = [os.path.abspath(f) for f in file_list]
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
        logging.info('Loading and preprocessing events, this might take a while')
        logging.debug('Loading events')
        with path(lookatthisgraph.resources, 'geo_array.npy') as p:
            file_inputs = [load_events(f, geo=p) for f in self.files]

        logging.debug('Preprocessing events')
        raw_pulses = [event['hits'] for finp in file_inputs for event in finp[0]]
        raw_truths = [event['params'] for finp in file_inputs for event in finp[0]]
        self.input_dict = {key: idx for idx, key in enumerate(file_inputs[0][1])}

        empty_mask = [i for i, ev in enumerate(raw_pulses) if len(ev) > 0]
        self.raw_pulses = [raw_pulses[i][:, :5] for i in empty_mask]
        self.raw_truths = [raw_truths[i] for i in empty_mask]

        self.n_events = len(self.raw_pulses)
        logging.info('%i events received' % self.n_events)

        logging.debug('Transforming truths')
        labels = ['zenith', 'energy', 'pid']
        self.transformed_truths = {label: self._transform_truths(label) for label in labels}

        logging.debug('Start normalizing pulses')
        self.normalized_features = self._get_normalized_features(normalization_parameters)
        logging.info('Data processing complete')


    def _get_normalized_features(self, normalization_parameters):
        features = [process_charges(event) for event in self.raw_pulses]
        pn = PulseNormalizer(features)
        features_normalized = pn.normalize('gauss', normalization_parameters)
        self.normalization_parameters = pn.get_normalization_parameters()
        return features_normalized


    def _transform_truths(self, label):
        if label == 'zenith':
            col = self.input_dict['zenith']
            transformed_truths = [[np.sin(y[col]), np.cos(y[col])] for y in self.raw_truths]

        elif label == 'energy':
            col = self.input_dict['neutrino_energy']
            transformed_truths = [[np.log10(y[col])] for y in self.raw_truths]

        elif label == 'pid':
            col = self.input_dict['track_energy']
            transformed_truths = [[1] if y[col] != 0 else [0] for y in self.raw_truths]

        else:
            raise ValueError('Truth label %s not recognized' % (label))

        return transformed_truths
