import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, knn_graph
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dropout_adj
from torch.nn import BatchNorm1d, PReLU
import torch_geometric.nn as NN

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.n_features = 5
        self.n_labels = 1
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
        self.drop = torch.nn.Dropout(.3)
        self.out = torch.nn.Linear(n_intermediate2, self.n_labels)
        self.out2 = torch.nn.Linear(self.n_labels, self.n_labels)


    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = knn_graph(x, 100, batch)
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

        x = self.drop(x)
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.linear4(x))
        x = F.leaky_relu(self.linear5(x))

        x = self.out(x)
        x = x.view(-1)

        return x
