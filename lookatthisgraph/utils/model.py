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
        """
        Class that holds network to evaluate data. Takes in config from Trainer.
        """
        self._source_dim = config['source_dim']
        self._target_dim = config['target_dim']
        self._classification = config['classification']
        self._knn_cols = config['knn_cols']
        self._normalize_output = config['normalize_output']
        self._final_activation = config['final_activation']
        print(self._final_activation)
        if config['net'] is not None:
            print('Use supplied network')
            self._net = config['net']
        else:
            print('Make network with %i input features, %i output clases' % (self._source_dim, self._target_dim))
            self._net = config['net_class'](self._source_dim,
                        self._target_dim,
                        self._knn_cols,
                        self._classification,
                        final_activation = self._final_activation,
                        normalize=self._normalize_output)
        self._device = torch.device(config['device']) if 'device' in config else torch.device('cuda')
        self._model = self._net.to(self._device)
        self._best_model = config['best_model'] if 'best_model' in config else None
        if self._best_model:
            self.load_best_model()


    def load_best_model(self):
        """
        Load parameters best model provided in config.
        Usually model with lowest validation error in training.
        """
        if self._device.type == 'cuda':
            self._model.load_state_dict(self._best_model)
            self._model.cuda()
        elif self._device.type == 'cpu':
            state_dict = deepcopy(self._best_model)
            for k, var_ in state_dict.items():
                state_dict[k] = var_.cpu()
            self._model.load_state_dict(state_dict)
            self._model.cpu()

    def set_device_type(self, device_type):
        """
        Change device used for evaluating

        Parameters:
        ----------
        device_type: str
            'cuda' (GPU) or 'cpu'
        """
        self._device = torch.device(device_type)
        if self._best_model:
            self.load_best_model()


    def evaluate_dataset(self, data, batch_size):
        """
        Evaluate all data in dataset or data list

        Parameters:
        ----------
        data: Dataset or list of Data
            Container of data with feature information
        batch_size: int
            Batch size used for evaluation
        """
        # data_list = dataset.data_list
        if isinstance(data, Dataset):
            data_list = data.data_list
        else:
            data_list = data
        loader = DataLoader(data_list, batch_size=batch_size)
        pred = evaluate_all(self._model, loader, self._device)
        pred = np.squeeze(pred.reshape(-1, self._target_dim))
        return pred
