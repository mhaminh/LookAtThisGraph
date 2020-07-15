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

    def __init__(self, file_list, normalization_parameters=None, logging_level=logging.INFO):
        if type(file_list) != list:
            raise TypeError('Input location has to be a list')
        self.files = [os.path.abspath(f) for f in file_list]
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging_level)
        logging.info('Loading and preprocessing events, this might take a while')

        logging.debug('Loading events')
        file_inputs = self._load_inputs()
        try:
            self._files_per_path = [len(np.load(os.path.join(p, 'subrun__categ_index.npy'))) for p in self.files]
            self.n_files = np.sum(self._files_per_path)
        except:
            logging.warning('Warning: Number of input files unknown')
        logging.debug('Preprocessing events')
        raw_pulses = [event['hits'] for finp in file_inputs for event in finp[0]]
        raw_truths = [event['params'] for finp in file_inputs for event in finp[0]]
        self.input_dict = {key: idx for idx, key in enumerate(file_inputs[0][1])}

        self.non_empty_mask = [i for i, ev in enumerate(raw_pulses) if len(ev) > 0]
        self.raw_pulses = [raw_pulses[i] for i in self.non_empty_mask]
        self.raw_truths = [raw_truths[i] for i in self.non_empty_mask]
        self.n_events = len(self.raw_pulses)
        logging.info('%i events received' % self.n_events)

        logging.debug('Transforming truths')
        labels = ['zenith', 'energy', 'pid']
        self.transformed_truths = {label: self._transform_truths(label) for label in labels}

        logging.debug('Start normalizing pulses')
        processed_features = self._preprocess_features()
        self.normalized_features = self._get_normalized_features(processed_features, normalization_parameters)
        logging.info('Data processing complete')

        self.filter = np.arange(len(self.normalized_features))
        self.filtered_features = self.normalized_features
        self.filtered_truths = self.transformed_truths

        self.results = {
            'zenith': None,
            'energy': None,
            'pid': None,
        }



    def _load_inputs(self):
        with path(lookatthisgraph.resources, 'geo_array.npy') as p:
            file_inputs = [load_events(f, geo=p) for f in self.files]
        return file_inputs


    def _preprocess_features(self):
        features = [process_charges(event[:, :5]) for event in self.raw_pulses]
        return features


    def _get_normalized_features(self, features, normalization_parameters):
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

    def apply_filter(self, filter_mask):
        self.filter = filter_mask
        try:
            self.filtered_features = self.normalized_features[filter_mask]
            self.filtered_truths = {key: item[filter_mask] for key, item in self.transformed_truths.items()}
        except:
            self.filtered_features = np.array([self.normalized_features[i] for i in filter_mask])
            self.filtered_truths = {key: np.array([item[i] for i in filter_mask]) for key, item in self.transformed_truths.items()}
        logging.info('Filter applied')

    def reset_filter(self):
        self.filter = np.arange(len(self.normalized_features))
        self.filtered_features = self.normalized_features
        self.filtered_truths = self.transformed_truths
        logging.info('Filter removed')


    def write_results(self, result, target_label):
        self.results[target_label] = np.array(result).flatten()
