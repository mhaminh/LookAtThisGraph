import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm.auto import tqdm
from copy import deepcopy
from torch_geometric.data import DataLoader, Data
from torch.nn import MSELoss, BCELoss
from lookatthisgraph.utils.model import Model
from lookatthisgraph.utils.datautils import build_data_list, evaluate_all, cart2polar, torch_to_numpy, calculate_splits
from lookatthisgraph.nets.ConvNet import ConvNet
from lookatthisgraph.utils.loss_functions import arc_loss, distance_loss
from lookatthisgraph.utils.torchutils import random_split
from torch.utils.data.sampler import WeightedRandomSampler


class Trainer:
    """
    Class for training from a Dataset.
    """
    def __init__(self, config):
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
        self.config = config
        self.dataset = config['dataset']
        force_new_datalist = config['force_new_datalist'] if 'force_new_datalist' in config else False
        if self.dataset.data_list is None or force_new_datalist:
            self.dataset.make_data_list(config['training_target'])
        self.data_list = self.dataset.data_list
        self._autosave_path = config['autosave_path'] if 'autosave_path' in config else None

        self._setup_network_info(config)
        self._model_config = self._make_model_config(config)
        self.model = Model(self._model_config)
        self._device = self.model._device
        self.net = self.model._net
        self.weight_calculator = config['weight_calculator'] if 'weight_calculator' in config else None
        # self._classification = config['classification'] if 'classification' in config else False
        # self._normalize_output = config['normalize_output'] if 'normalize_output' in config else False
        # self._device = torch.device('cuda') if 'device' not in config else torch.device(config['device'])

        if not (isinstance(self.crit, BCELoss)) and self.model._classification:
            logging.warning('Classification specified; did you provide the correct loss function? (BCELoss recommended)')
        elif (isinstance(self.crit, BCELoss)) and not self.model._classification:
            logging.warning('You provided a loss function for classifcation, but did not specify classification in config, are you sure?')

        logging.debug('Training using %d features on %d targets', self.model._source_dim, self.model._target_dim)
        # Initial permutation of data
        self.reshuffle()

        self._batch_size = config['batch_size']

        # Set configuration of training, validation, and testing samples
        train_split = config['train_split'] if 'train_split' in config else None
        test_split = config['test_split'] if 'test_split' in config else None
        val_split = config['validation_split'] if 'validation_split' in config else None
        self.splits = [train_split, val_split, test_split]

        # Setup dataloaders
        self.train_loader, self.val_loader, self.test_loader = self._get_loaders()
        self.test_truths = self.dataset.truths.compute().iloc[self.test_idx]
        self._sampling_weights = self.dataset.truths[config['weights']].copy().values if 'weights' in config else None

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['learning_rate'])
        if 'scheduling_step_size' in config and 'scheduling_gamma' in config:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['scheduling_step_size'],
                gamma=config['scheduling_gamma'])
        else:
            logging.info('No scheduler specified; use constant learning rate')

        self._plot = config['plot'] if 'plot' in config else False
        self._plot_filename = config['plot_filename'] if 'plot_filename' in config else 'training.pdf'

        self.train_losses = []
        self.validation_losses = []

        self.state_dicts = []
        self._max_epochs = config['max_epochs']

        self._fig, self._ax = None, None


    def _make_model_config(self, cfg):
        m_cfg = {}
        m_cfg['source_dim'] = self.data_list[0].x.shape[1]
        m_cfg['target_dim'] = len(self._target_col)
        m_cfg['classification'] = cfg['classification'] if 'classification' in cfg else False
        m_cfg['normalize_output'] = cfg['normalize_output'] if 'normalize_output' in cfg else False
        m_cfg['knn_cols'] = cfg['knn_cols'] if 'knn_cols' in cfg else [0, 1, 2, 3]
        m_cfg['device'] = 'cuda' if 'device' not in cfg else cfg['device']
        m_cfg['device'] = torch.device(m_cfg['device'])
        m_cfg['net_class'] = cfg['net_class'] if 'net_class' in cfg else ConvNet
        # m_cfg['device'] = torch.device('cuda') if 'device' not in cfg else torch.device(cfg['device'])
        m_cfg['final_activation'] = cfg['final_activation'] if 'final_activation' in cfg else None
        m_cfg['net'] = cfg['net'] if 'net' in cfg else None

        return m_cfg


    def reshuffle(self):
        """Reshuffle current permutation of data"""
        self.permutation = np.random.permutation(len(self.data_list))


    def _setup_network_info(self, config):
        """Setup dimensions, loss function etc. of network"""
        self.training_target = config['training_target']
        self._n_truths = len(self.data_list[0].y)
        self._target_col = [self.dataset.truth_cols[label] for label in self.training_target]
        # TODO: make more sophisticated, maybe in combination with feature label argument
        self._knn_cols = [0, 1, 2, 3]
        self.crit = config['loss_function']


    def get_new_loaders(self, train_split, val_split, test_split):
        """Make new dataloaders with supplied split ratios"""
        self.splits = [train_split, val_split, test_split]
        self.train_loader, self.val_loader, self.test_loader = self._get_loaders()


    def _get_loaders(self):
        """Calculates number of samples per loader and sets up dataloader"""
        n_train, n_val, n_test = calculate_splits(*self.splits, len(self.data_list))

        dataset_shuffled = [self.data_list[i] for i in self.permutation]

        self.train_idx = self.permutation[:n_train]
        self.val_idx = self.permutation[n_train:][:n_val]
        self.test_idx = self.permutation[n_train:][n_val:]

        if 'weights' in self.config:
            print('Using %s as weights' % (self.config['weights']))
            weights = self.dataset.truths[self.config['weights']].values
            train_sampler = WeightedRandomSampler(weights[self.train_idx], len(self.train_idx))
            weighted_val = self.config['weighted_validation'] if 'weighted_validation' in self.config else True
            if weighted_val:
                val_sampler = WeightedRandomSampler(weights[self.val_idx], len(self.val_idx))
            else:
                val_sampler = None
        else:
            train_sampler, val_sampler = None, None

        train_loader = DataLoader(dataset_shuffled[:n_train], self._batch_size, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(dataset_shuffled[n_train:n_train+n_val], self._batch_size, drop_last=True, sampler=val_sampler)
        test_loader = DataLoader(dataset_shuffled[n_train+n_val:][:n_test], self._batch_size, drop_last=True)

        return train_loader, val_loader, test_loader


    def train(self):
        """Start training model"""
        self._time_start = str(datetime.utcnow())
        self._train_perm = deepcopy(self.permutation)
        self.net.train()
        epoch_bar = tqdm(range(self._max_epochs), desc="Epochs")
        last_lr = float('inf')

        if self._plot:
            self._setup_plot()

        for epoch in epoch_bar:
            self.training_predictions = []
            self._train_epoch()
            self.state_dicts.append(deepcopy(self.net.state_dict()))
            self._val_epoch()

            epoch_bar.set_description("Train: %.2e, val: %.2e" % (self.train_losses[-1], self.validation_losses[-1]))
            if self._plot:
                self._plot_training()
            try:
                if self.scheduler.get_last_lr()[0] != last_lr:
                    last_lr = self.scheduler.get_last_lr()[0]
                    logging.info('Learning rate changed to %f in epoch %d', last_lr, epoch)

                self.scheduler.step()
            except AttributeError:
                pass
            logging.info("Training loss:%10.3e | Validation loss:%10.3e | Epoch %d / %d | Min validation loss:%10.3e at epoch %d",
                         self.train_losses[-1], self.validation_losses[-1], epoch, self._max_epochs, np.min(self.validation_losses), np.argmin(self.validation_losses))

            if self._autosave_path is not None:
                self.save_network_info(self._autosave_path)

        self._time_end = str(datetime.utcnow())


    def load_best_model(self):
        self.net.load_state_dict(self.state_dicts[np.argmin(self.validation_losses)])
        logging.info('Best model loaded')


    def _evaluate_loss(self, data):
        data = data.to(self._device)
        output = self.net(data)
        y = data.y.view(-1, self._n_truths)[:, self._target_col]
        label = y.to(self._device)
        loss = self.crit(output, label)
        return loss


    def _train_epoch(self):
        loss_all = 0
        for data in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            loss = self._evaluate_loss(data)
            loss.backward()
            loss_all += float(data.num_graphs * (loss.item()))
            self.optimizer.step()

        if np.isnan(loss_all):
            loss_all = float('inf')
        self.train_losses.append(loss_all / len(self.train_loader.dataset))


    def _val_epoch(self):
        with torch.no_grad():
            val_loss_all = 0
            for val_batch in self.val_loader:
                val_loss = self._evaluate_loss(val_batch)
                val_loss_all += float(val_batch.num_graphs * (val_loss.item()))
        if np.isnan(val_loss_all):
            val_loss_all = float('inf')
        self.validation_losses.append(val_loss_all / len(self.val_loader.dataset))


    def reweigh(self):
        # pred = evaluate_all(self.net, self.train_loader, 'cuda')
        pred = np.concatenate([torch_to_numpy(p) for p in self.training_predictions])
        weights = self.weight_calculator(pred, self.dataset.truths, len(self.train_losses))
#         _, z_pred = cart2polar(*pred.T)
#         edges = np.linspace(0, np.pi, 100)
#         binc_pred, _ = np.histogram(z_pred, edges, density=True)
#         if np.any(np.isclose(binc_pred, 0)):
#             print('Empty bins, skip boosting step')
#             pass
#         binc_truth, _ = np.histogram(self.dataset.truths['zenith'], edges, density=True)
#         idx = np.searchsorted(edges, self.dataset.truths['zenith']) - 1
#         weights = (binc_truth / binc_pred)[idx]
        if self._sampling_weights is None:
            self._sampling_weights = np.ones(self.dataset.truths.shape[0])
        # self._sampling_weights = weights
        self._sampling_weights *= weights
        train_sampler = torch.utils.data.WeightedRandomSampler(self._sampling_weights[self.train_idx], len(self.train_idx))
        # val_sampler = torch.utils.data.WeightedRandomSampler(self._sampling_weights[self.val_idx], len(self.val_idx))

        self.train_loader = DataLoader([self.data_list[i] for i in self.train_idx], self._batch_size, drop_last=True, sampler=train_sampler)
        # self.val_loader = DataLoader([self.data_list[i] for i in self.val_idx], self._batch_size, drop_last=True, sampler=val_sampler)

    def evaluate_test_samples(self):
        """Load best model and evaluate all test samples"""
        # self.load_best_model()
        return self.model.evaluate_dataset(self.test_loader.dataset, self.test_loader.batch_size)


    def save_network_info(self, location):
        """Save training info to pickle file"""
        training_info = {
            'file_names': self.dataset.files,
            'normalization_parameters': self.dataset.normalization_parameters,

            'source_dim': self.model._source_dim,
            'target_dim': self.model._target_dim,
            'classification': self.model._classification,
            'net': self.net,
            'normalize_output': self.model._normalize_output,
            'knn_cols': self.model._knn_cols,
            'final_activation': self.model._final_activation,

            'n_total': len(self.data_list),
            'n_train': len(self.train_loader.dataset),
            'n_val': len(self.val_loader.dataset),
            'n_test': len(self.test_loader.dataset),
            'train_idx': self.train_idx,
            'val_idx': self.val_idx,
            'test_idx': self.test_idx,
            'batch_size': self._batch_size,

            'training_target': self.training_target,
            'loss_function': str(self.crit),
            'permutation': self._train_perm,
            'optimizer': self.optimizer,

            'training_losses': self.train_losses,
            'validation_losses': self.validation_losses,
            'time_training_start': self._time_start,
            'best_model': self.state_dicts[np.argmin(self.validation_losses)],
        }
        try:
            training_info['time_training_end'] = self._time_end
        except AttributeError:
            pass
        try:
            training_info['scheduler'] = self.scheduler.state_dict()
        except AttributeError:
            pass

        pickle.dump(training_info, open(location, 'wb'))
        logging.info('Network dictionary saved')


    def _setup_plot(self):
        import matplotlib
        plot_backend = matplotlib.get_backend()
        if self._plot=='save':
            matplotlib.use('agg')
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._fig.show()
        self._fig.canvas.draw()
        matplotlib.use(plot_backend)


    def _plot_training(self):
        import matplotlib
        plot_backend = matplotlib.get_backend()
        if self._plot=='save':
            matplotlib.use('agg')
        self._ax.clear()
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.validation_losses, label="Validation")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        self._fig.canvas.draw()
        plt.pause(0.05)
        if self._plot == 'save':
            plt.savefig(self._plot_filename)
        matplotlib.use(plot_backend)
