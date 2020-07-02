import torch
from torch_geometric.data import DataLoader
import logging
from copy import deepcopy
from lookatthisgraph.nets.ConvNet import ConvNet
from lookatthisgraph.utils.datautils import build_data_list, evaluate


class Model:
    def __init__(self, config):
        self.training_target = config['training_target']
        self._target_dim = config['target_dim']
        self._classifcation = config['classification']
        self._include_charge = config['include_charge']
        self.net = config['model'] if 'model' in config else ConvNet(self._target_dim, self._classifcation)
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
    

    def evaluate_dataset(self, dataset, batch_size):
        data_list = build_data_list(
            dataset.normalized_features,
            dataset.transformed_truths[self.training_target],
            self._include_charge
        )
        loader = DataLoader(data_list, batch_size=batch_size, drop_last=True)
        pred, truth = evaluate(self.model, loader, self._device)
        pred = pred.reshape((-1, self._target_dim))
        truth = truth.reshape((-1, self._target_dim))
        return pred, truth
