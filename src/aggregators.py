import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(ABC, nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name):
        super().__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors):
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors):
        b = torch.max(neighbor_vectors, dim=-1).values
        c = b > 0.0
        d = c.float().sum(dim=-1, keepdim=True) + 1e-10
        e = d.repeat(1, 1, self.dim)
        neighbors_aggregated = neighbor_vectors.sum(dim=2) / e
        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=torch.relu, name=None):
        super().__init__(batch_size, dim, dropout, act, name)
        self.weights = nn.Parameter(torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _call(self, self_vectors, neighbor_vectors):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)
        output = self_vectors + neighbors_agg
        output = self.dropout(output)
        output = torch.matmul(output, self.weights) + self.bias
        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=torch.relu, name=None):
        super().__init__(batch_size, dim, dropout, act, name)
        self.weights = nn.Parameter(torch.randn(dim * 2, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _call(self, self_vectors, neighbor_vectors):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)
        output = torch.cat([self_vectors, neighbors_agg], dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output, self.weights) + self.bias
        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=torch.relu, name=None):
        super().__init__(batch_size, dim, dropout, act, name)
        self.weights = nn.Parameter(torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _call(self, self_vectors, neighbor_vectors):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors)
        output = neighbors_agg
        output = self.dropout(output)
        output = torch.matmul(output, self.weights) + self.bias
        return self.act(output)


class RoutingLayer(nn.Module):
    def __init__(self, layers, out_caps, cap_sz, batch_size, drop, inp_caps=None, name=None, tau=1.0):
        super().__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.batch_size = batch_size
        self.tau = tau
        self.drop = nn.Dropout(p=drop)

        self.cap_sz = cap_sz
        self.d, self.k = out_caps * cap_sz, out_caps

        if inp_caps is not None:
            self.inp_caps = inp_caps
            if layers == 1:
                self.w1 = nn.Parameter(torch.randn(inp_caps * cap_sz, cap_sz * out_caps) * (1. / torch.sqrt(inp_caps * cap_sz)))
                self.b1 = nn.Parameter(torch.randn(cap_sz * out_caps) * (1. / torch.sqrt(cap_sz * out_caps)))
            if layers == 2:
                self.w2 = nn.Parameter(torch.randn(inp_caps * cap_sz, cap_sz * out_caps) * (1. / torch.sqrt(inp_caps * cap_sz)))
                self.b2 = nn.Parameter(torch.randn(cap_sz * out_caps) * (1. / torch.sqrt(cap_sz * out_caps)))

    def forward(self, self_vectors, neighbor_vectors, max_iter):
        if hasattr(self, 'w1'):
            self_z = F.relu(torch.matmul(self_vectors.reshape(-1, self.inp_caps * self.cap_sz), self.w1) + self.b1)
            neighbor_z = F.relu(
                torch.matmul(neighbor_vectors.reshape(-1, self.inp_caps * self.cap_sz), self.w1) + self.b1)
        elif hasattr(self, 'w2'):
            self_z = F.relu(torch.matmul(self_vectors.reshape(-1, self.inp_caps * self.cap_sz), self.w2) + self.b2)
            neighbor_z = F.relu(
                torch.matmul(neighbor_vectors.reshape(-1, self.inp_caps * self.cap_sz), self.w2) + self.b2)
        else:
            self_z = self_vectors.reshape(-1, self.d)
            neighbor_z = neighbor_vectors.reshape(-1, self.d)

        self_z_n = F.normalize(self_z.reshape(self.batch_size, -1, self.k, self.d // self.k), dim=-1)
        neighbor_z_n = F.normalize(
            neighbor_z.reshape(self.batch_size, -1, neighbor_vectors.shape[-2], self.k, self.d // self.k), dim=-1)

        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = torch.zeros(self.batch_size, neighbor_vectors.shape[-3], neighbor_vectors.shape[-2], self.k).to(
                    self_z_n.device)
            else:
                p = torch.sum(neighbor_z_n * u.view(self.batch_size, -1, 1, self.k, self.d // self.k), dim=-1)
            p = F.softmax(p / self.tau, dim=-1)

            u = torch.sum(neighbor_z_n * p.view(self.batch_size, -1, neighbor_vectors.shape[-2], self.k, 1), dim=2)
            u += self_z_n
            if clus_iter < max_iter - 1:
                u = F.normalize(u, dim=-1)

        return self.drop(F.relu(u.reshape(self.batch_size, -1, self.d)))

