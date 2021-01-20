import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, knn_graph
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dropout_adj
from torch.nn import BatchNorm1d, PReLU
import torch_geometric.nn as NN

class ConvNet(torch.nn.Module):
    def __init__(self, n_features, n_labels, knn_cols, classification=False, normalize=False, final_activation=None):
        """
        Standard network architecture

        Parameters:
        ----------
        n_features: int
            Number of input features, i.e. dimension of input layer
        n_labels: int
            Number of prediction labels, i.e. dimension of output layer
        knn_cols: arr
            Column indices of features to be used for k-nearest-neighbor edge calculation
            Usually x, y, z, t
        classification: bool
            Switches to classification loss
        normalize: bool
            Whether to normalize ouput (e.g. for prediction of vector on unit sphere)
        """
        super(ConvNet, self).__init__()
        self.classification = classification
        self._normalize = normalize
        self._knn_cols = knn_cols
        if normalize == True and classification == True:
            print("Warning: \'normalize\' not defined for \'classfication\', will be ignored")
        self.n_features = n_features
        self.n_labels = n_labels
        n_intermediate = 128
        n_intermediate2 = 6*n_intermediate
        self.conv1 = TAGConv(self.n_features, n_intermediate, 2)
        self.conv2 = TAGConv(n_intermediate, n_intermediate, 2)
        self.conv3 = TAGConv(n_intermediate, n_intermediate, 2)
        ratio = .9
        self.batchnorm1 = BatchNorm1d(n_intermediate2)
        self.linear1 = torch.nn.Linear(n_intermediate2, n_intermediate2)
        self.linear2 = torch.nn.Linear(n_intermediate2, n_intermediate2)
        self.linear3 = torch.nn.Linear(n_intermediate2, n_intermediate2)
        self.linear4 = torch.nn.Linear(n_intermediate2, n_intermediate2)
        self.linear5 = torch.nn.Linear(n_intermediate2, n_intermediate2)
        dropout_ratio = .3
        self.drop1 = torch.nn.Dropout(.3)
        self.drop2 = torch.nn.Dropout(.3)
        self.drop3 = torch.nn.Dropout(.3)
        self.drop4 = torch.nn.Dropout(.3)
        self.drop5 = torch.nn.Dropout(.3)
        self.out = torch.nn.Linear(n_intermediate2, self.n_labels)
        self.final_activation = final_activation


    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = knn_graph(x[:, self._knn_cols], 80, batch)
        edge_index, _ = dropout_adj(edge_index, p=0.3)
        batch = data.batch

        x = F.leaky_relu(self.conv1(x, edge_index))
        x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.leaky_relu(self.conv2(x, edge_index))
        x2 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.leaky_relu(self.conv3(x, edge_index))
        x3 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.batchnorm1(x)

        x = F.leaky_relu(self.linear1(x))

        x = self.drop1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.drop2(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.drop3(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.drop4(x)
        x = F.leaky_relu(self.linear5(x))
        x = self.drop5(x)

        x = self.out(x)
        if self.classification:
            x = torch.sigmoid(x)
        elif self._normalize:
            x = x.view(-1, self.n_labels)
            norm = torch.norm(x, dim=1).view(-1, 1)
            x = x / norm
        elif self.final_activation is not None:
            x = self.final_activation(x)

        x = x.view(-1, self.n_labels)

        return x
