import numpy as np
from tqdm.auto import tqdm

class PulseNormalizer:
    '''
    Takes pulses from dataloader.get_pulses and processes them for NN input
    Minmax normalizer

    Parameters:
    ----------
    events: list
        List with arrays of pulse information
    norm_cols: list
        Indices of columns to normalize
    '''
    def __init__(self, events, norm_cols=None, eps=None):
        self._events = events
        self._norm_cols = norm_cols if norm_cols else list(range(events[0].shape[1]))

        self.mode = None
        self._linear_parameters = self._get_linear_parameters()
        self._norm_parameters = self._get_norm_parameters()

        self.eps = eps # Tuple with column and eps to filter out


    def _get_min_max(self):
        minmax = []
        for col in self._norm_cols:
            features = np.concatenate([event[:, col] for event in self._events])
            mm = [np.min(features), np.max(features)]
            minmax.append(mm)

        return minmax


    def _get_linear_parameters(self):
        minmaxs = self._get_min_max()
        pars = [[mm[1]-mm[0], mm[0]] for mm in minmaxs]
        return np.asarray(pars)


    def _get_norm_parameters(self):
        norm_parameters = []
        for col in self._norm_cols:
            features = np.concatenate([ev[:, col] for ev in self._events])
            pars = [np.mean(features), np.std(features)]
            norm_parameters.append(pars)
        return np.asarray(norm_parameters)


    def _normalize_event(self, features):
        if self.mode == 'minmax':
            return (features - self._linear_parameters[:,1]) / self._linear_parameters[:,0]
        elif self.mode == 'gauss':
            return (features - self._norm_parameters[:, 0]) / self._norm_parameters[:,1]

    def _get_normalized_event(self, event):
        new_event = np.copy(event)
        features = new_event[:, self._norm_cols]
        norm_feat = self._normalize_event(features)
        new_event[:, self._norm_cols] = norm_feat
        return new_event


    def normalize(self, mode):
        self.mode = mode
        nn = [self._get_normalized_event(event) for event in tqdm(self._events, desc='Normalizing events')]
        return nn

