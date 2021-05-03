import torch
import logging
from torch_geometric.data import Data
from tqdm.auto import tqdm
import pandas as pd
import dask
import dask.dataframe as dd
import numpy as np
import os
from joblib import Parallel, delayed
from lookatthisgraph.utils.i3cols_to_pandas import get_pulses, get_truths, get_energies, dataframe_to_event_list
from lookatthisgraph.utils.dataloader import get_pulses as get_pulses_hf5
from lookatthisgraph.utils.icecubeutils import get_dom_positions
from lookatthisgraph.utils.datautils import flatten


class Dataset:
    # TODO: maybe add option to select which pulse frame to use (SRT TW cleaned pulses most usual though)
    # TODO: add option to choose which features and truths to use
    # TODO: maybe add option to move normalization and datalist creation to trainer
    # TODO: add filter
    # TODO: function to combine Datasets
    # TODO: add functions to set features / truths
    # TODO: Check if resetting indices of dataframes is needed in converter
    def __init__(self,
            indir_list,
            upgrade=False,
            feature_labels=None,
            truth_labels=['x', 'y', 'z', 'x_dir', 'y_dir', 'z_dir', 'log10(energy)', 'log10(shower_energy)', 'log10(track_energy)', 'PID'],
            save_energies=False,
            force_recalculate=False,
            make_data_list=False):
        self.files = indir_list
        self.upgrade = upgrade
        if feature_labels is None and not self.upgrade:
            self.feature_labels = ['x_om', 'y_om', 'z_om', 'time', 'charge']
        elif feature_labels is None and self.upgrade:
            self.feature_labels = ['x_om', 'y_om', 'z_om', 'time', 'charge', 'xdir_om', 'ydir_om', 'zdir_om', 'is_IceCube', 'is_PDOM', 'is_mDOM', 'is_DEgg']
        logging.info('Loading inputs')
        self.raw_pulses, self.truths = self._load_inputs(save_energies, force_recalculate)
        event_idx = list(self.raw_pulses['event'].unique())
        self._non_empty_mask = self.truths['event'].isin(event_idx) # Remove truths of empty pulses
        self.truths = self.truths[self._non_empty_mask]
        self.raw_pulses.reset_index(drop=True)
        self.truths.reset_index(drop=True)
        self._means, self._stds = self._get_normalization_parameters()
        self.normalized_pulses = None
        logging.info('Normalizing pulses')
        self._make_normalized_pulses(self._means, self._stds)
        self.normalization_parameters = {'means':  self._means, 'stds': self._stds}
        self.data_list = None
        self.n_events = None
        if make_data_list:
            logging.info('Making data list')
            self.make_data_list(truth_labels)

    def _load_inputs(self, save_energies=False, force_recalculate=False):
        """
        Load events and truths from input directories
        Increment event index based on files loaded
        """
        if not isinstance(self.files, list):
            raise TypeError('Input directories have to be list')
        elif len(self.files) == 0:
            raise ValueError('Input list empty')
        elif len(self.files) == 1:
            indir = self.files[0]
            return get_pulses(indir, self.upgrade), get_truths(indir, save_energies=save_energies, force_recalculate=force_recalculate)
        # TODO: Make index checks
        else:
            events = [get_pulses(indir, self.upgrade) for indir in self.files]
            truths = [get_truths(indir,
                                 save_energies=save_energies,
                                 force_recalculate=force_recalculate)
                    for indir in self.files]
            # Increment event indices
            last_event_idx = np.array([np.max(np.unique(frame['event'])) for frame in events])
            increments = np.cumsum(np.concatenate([[0], (np.array(last_event_idx[:-1])+1)]))
            for event_frame, truth_frame, inc in zip(events, truths, increments):
                event_frame['event'] += inc
                truth_frame['event'] += inc
            events = dd.concat(events, ignore_index=True, sort=False)
            truths = dd.concat(truths, ignore_index=True, sort=False)
            return events, truths

    def _get_normalization_parameters(self):
        df = self.raw_pulses[self.feature_labels]
        if 'omtype' in df.columns:
            df = df.drop('omtype', axis=1)
        means = df.mean()
        stds = df.std()
        return means, stds

    def _make_normalized_pulses(self, means, stds):
        # TODO: add option to exclude/include certain columns; make label checks
        # Maybe only normalize in Trainer? Saves memory
        event_idx = self.raw_pulses['event'].copy()
        df = self.raw_pulses[self.feature_labels]
        if 'omtype' in df.columns:
            df = df.drop('omtype', axis=1)
        normalized = (df - means) / stds
        normalized = dd.concat([normalized, event_idx], axis=1)
        self.normalized_pulses = normalized

    def renormalize(self, means, stds):
        self._make_normalized_pulses(means, stds)
        self.make_data_list()

    def make_data_list(self, truth_labels=None):
        event_list = dataframe_to_event_list(self.normalized_pulses)
        if truth_labels is None:
            truth_labels = self.truths.columns
        truths = self.truths[truth_labels]
        truths = truths.compute()
        truths = truths.values
        difference = np.setdiff1d(self.normalized_pulses['event'].unique(), self.truths['event'].unique())
        if len(difference) > 0:
            raise ValueError('Number of entries in event list and truths not matching: Events %i' % (difference))
        data_list = [Data(x=torch.tensor(x, dtype=torch.float),
                          y=torch.tensor(y, dtype=torch.float)) for x, y in tqdm(zip(event_list, truths), total=len(event_list), desc='Making event list')]
        self.data_list = data_list
        self.n_events = len(data_list)
        self.truth_cols = {truth_label: i for i, truth_label in enumerate(truth_labels)}

#     def set_weights(self, weights):
#         if 'weight' in self.truths:
#             self.truths['weight'] == weights
#         else:
#             w = pd.Series(weights, name='weight')
#             self.truths = pd.concat([self.truths, w], axis=1)
#         self.make_data_list()

    def add_truth(self, df, overwrite=False):
        intersect = np.intersect1d(df.columns, self.truths.columns)
        if len(intersect)!=0 and not overwrite:
            raise IndexError('Names already exist:', intersect, '; Choose different column names')
        elif overwrite:
            for col in intersect:
                del self.truths[col]
        self.truths = pd.concat([self.truths, df], axis=1)

    def save(self, fname, overwrite=False):
        if os.path.exists(fname) and not overwrite:
            raise FileError('File already exists')
        elif overwrite:
            logging.info('Overwriting existing file')
            os.remove(fname)
        with pd.HDFStore(fname) as s:
            s['data/pulses'] = self.raw_pulses
            s['data/truths'] = self.truths
            s['normalization_parameters/means'] = self._means
            s['normalization_parameters/stds'] = self._stds
            s['data/is_upgrade'] = pd.Series(self.upgrade, name='is_upgrade')


class DatasetFromSave(Dataset):
    def __init__(self,
            fname,
            feature_labels=None,
            truth_labels=['x', 'y', 'z', 'x_dir', 'y_dir', 'z_dir', 'log10(energy)', 'log10(shower_energy)', 'log10(track_energy)', 'PID'],
            make_data_list=True):
        with pd.HDFStore(fname) as s:
            self.raw_pulses = s['data/pulses']
            self.truths = s['data/truths']
            self._means = s['normalization_parameters/means']
            self._stds = s['normalization_parameters/stds']
            self.upgrade = bool(s['data/is_upgrade'].values)

        self.feature_labels = feature_labels
        if feature_labels is None and not self.upgrade:
            self.feature_labels = ['x_om', 'y_om', 'z_om', 'time', 'charge']
        elif feature_labels is None and self.upgrade:
            self.feature_labels = ['x_om', 'y_om', 'z_om', 'time', 'charge', 'xdir_om', 'ydir_om', 'zdir_om', 'is_IceCube', 'is_PDOM', 'is_mDOM', 'is_DEgg']
        self.truth_labels = truth_labels
        self.normalized_pulses = None
        self._make_normalized_pulses(self._means, self._stds)
        self.n_events = None
        self.make_data_list()
