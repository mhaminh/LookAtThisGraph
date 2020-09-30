import os
import logging
from importlib.resources import path
import numpy as np
# from joblib import Parallel, delayed
import torch
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import lookatthisgraph.resources
from torch_geometric.data import Data, DataLoader
from lookatthisgraph.utils.datautils import process_charges, build_data_list, truths_to_array, flatten
from lookatthisgraph.utils.pulsenormalizer import PulseNormalizer
from lookatthisgraph.utils.i3cols_dataloader import load_events
from lookatthisgraph.utils.dataloader import get_pulses
from lookatthisgraph.utils.icecubeutils import get_dom_positions


class Dataset(object):

    """Dataset for training of neural net

    Loads in data converted from i3cols. Normalizes pulses and transforms truths to be compatible with network. Fills list with PyTorch Data objects for later handling.

    Attributes:
    ----------
    raw_pulses: list
        Raw pulses from input file, empty events filtered out
    raw_truths: array
        Raw truths from input file, empty events filtered out
    input_dict: dict
        Describes which truth quantity is in which column
    non_empty_mask: list
        Indices of non-empty events
    normalized_features: list
        Normalized features
    transformed_truths: dict
        Transformed truths, ordered by labels
    filter: list
        Describes which events are used for training / evaluating
    filtered_features: list
        Normalized input features with filter applied
    filtered_truths: dict
        Transformed truths with filter applied
    data_list: list
        List of PyTorch Data objects for training / evaluating
    truth_cols: array
        Describes column indices of truths in data_list
    results: dict
        For storing reconstruced events for later conversion to PISA files
    """

    def __init__(self, file_list, normalization_parameters=None, logging_level=logging.INFO, fill_list=True):
        if type(file_list) != list:
            raise TypeError('Input location has to be a list')
        self.files = [os.path.abspath(f) for f in file_list]
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging_level)
        logging.info('Loading and preprocessing events, this might take a while')

        try:
            self._files_per_path = [len(np.load(os.path.join(p, 'subrun__categ_index.npy'))) for p in self.files]
            self.n_files = np.sum(self._files_per_path)
        except:
            logging.warning('Warning: Number of input files unknown')
        logging.debug('Preprocessing events')

        logging.debug('Loading events')
        raw_pulses, raw_truths, self.input_dict = self._load_inputs()

        # Filter out empty events
        self.non_empty_mask = [i for i, ev in enumerate(raw_pulses) if len(ev) > 0]
        self.raw_pulses = [raw_pulses[i] for i in self.non_empty_mask]
        self.raw_truths = [raw_truths[i] for i in self.non_empty_mask]
        self.n_events = len(self.raw_pulses)
        logging.info('%i events received' % self.n_events)

        logging.debug('Transforming truths')
        labels = ['zenith', 'energy', 'pid', 'dir_cart']
        self.transformed_truths = {label: self._transform_truths(label) for label in labels}

        logging.debug('Start normalizing pulses')
        processed_features = self._preprocess_features()
        self.normalized_features = self._get_normalized_features(processed_features, normalization_parameters)
        logging.info('Data processing complete')

        self.filter = np.arange(len(self.normalized_features))
        self.filtered_features = self.normalized_features
        self.filtered_truths = self.transformed_truths

        self.truth_cols, truth_arr = truths_to_array(self.filtered_truths)

        if fill_list:
            self._raw_data_list = build_data_list(
                self.filtered_features,
                truth_arr,
            )
            self.data_list = self._raw_data_list

        self.results = {
            'zenith': None,
            'energy': None,
            'pid': None,
        }



    def _load_inputs(self):
        """Loads data from i3cols output"""
        with path(lookatthisgraph.resources, 'geo_array.npy') as p:
            file_inputs = [load_events(f, geo=p) for f in self.files]
        raw_pulses = [event['hits'] for finp in file_inputs for event in finp[0]]
        raw_truths = [event['params'] for finp in file_inputs for event in finp[0]]
        input_dict = {key: idx for idx, key in enumerate(file_inputs[0][1])}
        return raw_pulses, raw_truths, input_dict


    def _preprocess_features(self):
        """Converts charges (if necessary) to log(charges)

        Network seems to handle log(charges) better
        """
        features = [process_charges(event[:, :5]) for event in self.raw_pulses]
        return features


    def _get_normalized_features(self, features, normalization_parameters):
        """Normalize input features
        Parameters:
        ----------
        features: list of arrays
            List of input features
        normalization_parameters: array, optional
            Normalization parameters per input column, provide if using trained model, gets calculated otherwise.

        Returns:
        ----------
        features_normalized: list of arrays
            Normalized features
        """
        pn = PulseNormalizer(features)
        features_normalized = pn.normalize('gauss', normalization_parameters)
        self.normalization_parameters = pn.get_normalization_parameters()
        return features_normalized


    def _transform_truths(self, label):
        """Transform input truths depending on quantity
        Networks seem to handle these transformed values better
        """
        if label == 'zenith':
            col = self.input_dict['zenith']
            # Transform to 2D cartesian
            transformed_truths = [[np.sin(y[col]), np.cos(y[col])] for y in self.raw_truths]

        elif label == 'energy':
            col = self.input_dict['neutrino_energy']
            transformed_truths = [[np.log10(y[col])] for y in self.raw_truths]

        elif label == 'pid':
            col = self.input_dict['track_energy']
            # Event is track if track_energy > 0
            transformed_truths = [[1] if y[col] != 0 else [0] for y in self.raw_truths]

        elif label == 'dir_cart':
            col_a = self.input_dict['azimuth']
            col_z = self.input_dict['zenith']
            # Transform to 3D cartesian
            transformed_truths = [
                [np.sin(y[col_z]) * np.cos(y[col_a]),
                np.sin(y[col_z]) * np.sin(y[col_a]),
                np.cos(y[col_z])]
                    for y in self.raw_truths]

        else:
            raise ValueError('Truth label %s not recognized' % (label))

        return np.array(transformed_truths)


    def apply_filter(self, filter_mask):
        """Apply / Replace current filter

        Affects which values are used for evaluation / training.

        Parameters:
        ----------
        filter_mask: array
            Can be indices (e.g. [1, 2, 4, 8, ...]) or boolean (e.g. [True, True, False, True, ...])
        """
        self.filter = filter_mask
        try:
            self.filtered_features = self.normalized_features[filter_mask]
            self.filtered_truths = {key: item[filter_mask] for key, item in self.transformed_truths.items()}
            self.data_list = self._raw_data_list[filter_mask]
        except:
            self.filtered_features = np.array([self.normalized_features[i] for i in filter_mask])
            self.filtered_truths = {key: np.array([item[i] for i in filter_mask]) for key, item in self.transformed_truths.items()}
            self.data_list = [self._raw_data_list[i] for i in filter_mask]
        logging.info('Filter applied')


    def reset_filter(self):
        """Reset current filter"""
        self.filter = np.arange(len(self.normalized_features))
        self.filtered_features = self.normalized_features
        self.filtered_truths = self.transformed_truths
        self.data_list = self._raw_data_list
        logging.info('Filter removed')


    def renormalize(self, normalization_parameters):
        """Renormalize current dataset

        Could be done easier.
        """
        processed_features = self._preprocess_features()
        self.normalized_features = self._get_normalized_features(processed_features, normalization_parameters)

        self.apply_filter(self.filter)

        self.truth_cols, truth_arr = truths_to_array(self.filtered_truths)

        self._raw_data_list = build_data_list(
            self.filtered_features,
            truth_arr,
        )
        self.data_list = self._raw_data_list
        self.apply_filter(self.filter)


    def write_results(self, result, target_label):
        self.results[target_label] = np.array(result).flatten()
