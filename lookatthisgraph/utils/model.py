import torch
import numpy as np
from torch_geometric.data import DataLoader
import logging
from copy import deepcopy
from lookatthisgraph.nets.ConvNet import ConvNet
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.datautils import build_data_list, evaluate_all


class Model:
    def __init__(self, config):
        self.training_target = config['training_target']
        self.n_features = config['source_dim']
        self._target_dim = config['target_dim']
        self._classifcation = config['classification']
        self._knn_cols = config['knn_cols']
        self._normalize_output = config['normalize_output']
        # self._include_charge = config['include_charge']
        self.net = config['model'] if 'model' in config else ConvNet(self.n_features, self._target_dim, self._knn_cols, self._classifcation, normalize=self._normalize_output)
        self._device = torch.device(config['device']) if 'device' in config else torch.device('cuda')
        self.model = self.net.to(self._device)
        self._best_model = config['best_model'] if 'best_model' in config else None
        self.load_best_model() if self._best_model is not None else None


    def load_best_model(self):
        if self._device.type == 'cuda':
            self.model.load_state_dict(self._best_model)
            self.model.cuda()
        elif self._device.type == 'cpu':
            state_dict = deepcopy(self._best_model)
            for k, v in state_dict.items():
                  state_dict[k] = v.cpu()
            self.model.load_state_dict(state_dict)
            self.model.cpu()

    def set_device_type(self, device_type):
        self._device = torch.device(device_type)
        self.load_best_model() if self._best_model is not None else 0


    def evaluate_dataset(self, data, batch_size):
        """
        Evaluate all data in dataset or data list

        # data_list = dataset.data_list
        if isinstance(data, Dataset):
            data_list = data.data_list
        else:
            data_list = data
        loader = DataLoader(data_list, batch_size=batch_size)
        pred = evaluate_all(self._model, loader, self._device)
        pred = np.squeeze(pred.reshape(-1, self._target_dim))
        return pred
