import torch
import numpy as np
from torch_geometric.data import DataLoader
import logging
from copy import deepcopy
from lookatthisgraph.nets.ConvNet import ConvNet
from lookatthisgraph.utils.datautils import build_data_list, evaluate


class Model:
    def __init__(self, config):
        self.training_target = config['training_target']
        self.n_features = config['source_dim']
        self._target_dim = config['target_dim']
        self._classifcation = config['classification']
        # self._include_charge = config['include_charge']
        self.net = config['model'] if 'model' in config else ConvNet(self.n_features, self._target_dim, self._classifcation)
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


    # TODO: use evaluate_all method, don't return truths
    def evaluate_dataset(self, dataset, batch_size, evaluate_all=True):
        data_list = dataset.data_list
        n_rest = len(data_list) % batch_size
        if len(data_list) >= batch_size:
            loader = DataLoader(data_list[:-n_rest], batch_size=batch_size)
            pred = evaluate(self.model, loader, self._device, mode='eval')
            pred = (pred.reshape(-1, self._target_dim))
        else:
            pred = np.empty((0, self._target_dim))

        if evaluate_all:
            rest_loader = DataLoader(data_list[-n_rest:], batch_size=n_rest)
            pred_rest = evaluate(self.model, rest_loader, self._device, mode='eval', pbar=False)
            pred_rest = pred_rest.reshape(-1, self._target_dim)

            pred = np.concatenate([pred, pred_rest])
        truth = np.array([np.array(d.y) for d in data_list])[:len(pred)]
        truth = {key: truth[:, cols] for key, cols in dataset.truth_cols.items()}
        return pred, truth
