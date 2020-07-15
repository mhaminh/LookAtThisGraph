import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
from tqdm.auto import tqdm
from copy import deepcopy
from torch_geometric.data import DataLoader
from torch.nn import MSELoss, BCELoss
from lookatthisgraph.utils.datautils import build_data_list, evaluate
from lookatthisgraph.nets.ConvNet import ConvNet


class Trainer:
    def __init__(self, config):
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
        self.dataset = config['dataset']
        self.training_target = config['training_target']
        target = self.dataset.filtered_truths[self.training_target]
        self.include_charge = config['include_charge'] if 'include_charge' in config else True
        self.data_list = build_data_list(
            self.dataset.filtered_features,
            target,
            self.include_charge
        )
        self._target_dim = len(self.data_list[0].y)
        self._source_dim = self.data_list[0].x.shape[1]
        self.reshuffle()

        self._batch_size = config['batch_size']

        self._train_split = config['train_split'] if 'train_split' in config else None
        self._test_split = config['test_split'] if 'test_split' in config else None
        self._val_split = config['validation_split'] if 'validation_split' in config else 'batch'

        self.train_loader, self.val_loader, self.test_loader = self._get_loaders()

        if 'loss_function' not in config:
            self.crit = BCELoss() if self.training_target == 'pid' else MSELoss()
        self._classification = bool(isinstance(self.crit, BCELoss))

        self._device = torch.device('cuda') if 'device' not in config else torch.device(config['device'])
        net = config['net'] if 'net' in config else ConvNet(self._source_dim, self._target_dim, self._classification)
        self.model = net.to(self._device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        if 'scheduling_step_size' in config and 'scheduling_gamma' in config:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['scheduling_step_size'],
                gamma=config['scheduling_gamma'])
        else:
            logging.info('No scheduler specified; use constant learning rate')

        self._plot = config['plot'] if 'plot' in config else False
        if self._plot == 'save':
            plt.switch_backend('agg')

        self.train_losses = []
        self.validation_losses = []

        self.state_dicts = []
        self._max_epochs = config['max_epochs']

        self._fig, self._ax = None, None

    def reshuffle(self):
        self.permutation = np.random.permutation(len(self.data_list))


    def _get_loaders(self):
        split = lambda s: int(self.dataset.n_events * s) if s < 1 else int(s)

        if self._val_split == 'batch':
            n_val = self._batch_size
        else:
            n_val = split(self._val_split)
        if self._train_split is None:
            if self._test_split is not None:
                n_test = split(self._test_split)
            else:
                n_test = 0
            n_train = len(self.data_list) - n_val - n_test
        else:
            n_train = split(self._train_split)
            if self._test_split is not None:
                n_test = split(self._test_split)
            else:
                n_test = len(self.data_list) - n_train - n_val

        logging.info('%d training samples, %d validation samples, %d test samples received; %d ununsed',
                n_train, n_val, n_test, len(self.data_list) - n_train - n_val - n_test)
        if n_train + n_val + n_test > self.dataset.n_events:
            raise ValueError('Loader configuration exceeds number of data samples')

        dataset_shuffled = [self.data_list[i] for i in self.permutation]

        train_loader = DataLoader(dataset_shuffled[:n_train], self._batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(dataset_shuffled[n_train:n_train+n_val], self._batch_size, drop_last=True)
        test_loader = DataLoader(dataset_shuffled[n_train+n_val:][:n_test], self._batch_size, drop_last=True)

        return train_loader, val_loader, test_loader


    def train(self):
        import matplotlib.pyplot as plt
        self._time_start = str(datetime.utcnow())
        self._train_perm = deepcopy(self.permutation)
        self.model.train()
        epoch_bar = tqdm(range(self._max_epochs))
        last_lr = float('inf')

        if self._plot != False :
            self._setup_plot()

        for epoch in epoch_bar:

            self._train_epoch()
            self.state_dicts.append(self.model.state_dict())
            self._val_epoch()

            epoch_bar.set_description("Train: %.2e, val: %.2e" % (self.train_losses[-1], self.validation_losses[-1]))
            if self._plot:
                self._plot_training()
            try:
                if self.scheduler.get_lr()[0] != last_lr:
                    last_lr = self.scheduler.get_lr()[0]
                    logging.info('Learning rate changed to %f in epoch %d', last_lr, epoch)

                self.scheduler.step()
            except AttributeError:
                pass
            logging.info("Training loss:%10.3e | Validation loss:%10.3e | Epoch %d / %d | Min validation loss:%10.3e at epoch %d",
                         self.train_losses[-1], self.validation_losses[-1], epoch, self._max_epochs, np.min(self.validation_losses), np.argmin(self.validation_losses))

        self._time_end = str(datetime.utcnow())


    def load_best_model(self):
        self.model.load_state_dict(self.state_dicts[np.argmin(self.validation_losses)])
        logging.info('Best model loaded')


    def save_best_model(self, location):
        torch.save(self.state_dicts[np.argmin(self.validation_losses)], location)
        logging.info('Best model saved')


    def _train_epoch(self):
        loss_all = 0
        for data in self.train_loader:
            data = data.to(self._device)
            self.optimizer.zero_grad()
            output = self.model(data)
            label = data.y.to(self._device)
            loss = self.crit(output, label)
            loss.backward()
            loss_all += float(data.num_graphs * (loss.item()))
            self.optimizer.step()

        self.train_losses.append(loss_all / len(self.train_loader.dataset))


    def _val_epoch(self):
        with torch.no_grad():
            val_loss_all = 0
            for val_batch in self.val_loader:
                val_data = val_batch.to(self._device)
                out_val = self.model(val_data)
                val_loss = self.crit(out_val, val_data.y)
                val_loss_all += float(val_data.num_graphs * (val_loss.item()))
        self.validation_losses.append(val_loss_all / len(self.val_loader.dataset))


    def evaluate_test_samples(self):
        pred, truth = evaluate(self.model, self.test_loader, self._device)
        if self._target_dim > 1:
            pred = pred.reshape((-1, self._target_dim))
            truth = truth.reshape((-1, self._target_dim))
        else:
            pred, truth = pred.flatten(), truth.flatten()
        return pred, truth


    def save_network_info(self, location):
        training_info = {
            'file_names': self.dataset.files,
            'training_target': self.training_target,
            'include_charge': self.include_charge,
            'source_dim': self._source_dim,
            'target_dim': self._target_dim,
            'classification': self._classification,
            'n_total': len(self.data_list),
            'n_train': len(self.train_loader.dataset),
            'n_val': len(self.val_loader.dataset),
            'n_test': len(self.test_loader.dataset),
            'batch_size': self._batch_size,
            'normalization_parameters': self.dataset.normalization_parameters,
            'loss_function': str(self.crit),
            'net': self.model,
            'optimizer': self.optimizer,
            'training_losses': self.train_losses,
            'validation_losses': self.validation_losses,
            'time_training_start': self._time_start,
            'permutation': self._train_perm,
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

        return training_info


    def _setup_plot(self):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._fig.show()
        self._fig.canvas.draw()


    def _plot_training(self):
        self._ax.clear()
        plt.plot(self.train_losses, label="Training")
        plt.plot(self.validation_losses, label="Validation")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        self._fig.canvas.draw()
        plt.pause(0.05)
        if self._plot == 'save':
            plt.savefig('training.pdf')
