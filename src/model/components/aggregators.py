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
    def __init__(
            self, layers, out_caps, nhidden, drop,
            inp_caps=None, name=None, tau=1.0, edge_feature_dim=0, lora_edge_feats=False
    ):
        super().__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.tau = tau
        self.drop = nn.Dropout(p=drop)

        self.nhidden = nhidden
        self.k = out_caps
        self.inp_caps = inp_caps

        if self.inp_caps is not None:
            if layers == 1:
                self.fc1 = nn.Linear(self.inp_caps * nhidden, nhidden * self.k)
            if layers == 2:
                self.fc2 = nn.Linear(self.inp_caps * nhidden, nhidden * self.k)

        # idea for edge features:
        # given current code, it needs to modify neighbor_z,
        # but I cant change the fc weights, because they are shared for processing the self embedding
        # Also, I cant change the dimensionality of the neighbor_z, because that should match the self_z
        #
        # Now, 2 main options that i see:
        # 1. have a layer (n_edge_features -> self.inp_caps * self.nhidden)
        #   and then add the output to the corresponding neighbor_z.
        #   Downside is that it will not take into account the node features
        # 2. Have a LoRA that takes neighbor_z concatenated with edge_features and outputs shape like neighbor_z
        #   then add it to the neighbor_z
        self.edge_feature_dim = edge_feature_dim
        self.lora_edge_feats = lora_edge_feats
        if self.edge_feature_dim > 0:
            if not self.lora_edge_feats:
                self.edge_fc = nn.Linear(self.edge_feature_dim, self.nhidden * self.k)
            else:
                raise NotImplementedError("LoRA with edge features is not implemented yet")

    def forward(self, self_vectors, neighbor_vectors, max_iter, edge_feats=None):
        """

        :param self_vectors: [batch_size, num_considered_nodes, k * n_hidden].
            num_considered_nodes starts with 1,
            then goes to args.user_neighbor or args.news_neighbor for news and users respectively
        :param neighbor_vectors: [batch_size, num_considered_nodes, n_neighbors, k * n_hidden]
        :param max_iter: int
        :param edge_feats: [batch_size, num_considered_nodes, n_neighbors, edge_feature_dim].
            Edge features for neighbors
        """
        batch_size = self_vectors.shape[0]
        num_considered_nodes = self_vectors.shape[-2]
        n_neighbors = neighbor_vectors.shape[-2]
        self_vectors = self.drop(self_vectors)
        neighbor_vectors = self.drop(neighbor_vectors)

        if hasattr(self, 'fc1'):
            # this reshape converts to [all_nodes, their_full_embedding_size]
            self_z = F.relu(self.fc1(self_vectors.reshape(-1, self.inp_caps * self.nhidden)))
            neighbor_z = F.relu(self.fc1(neighbor_vectors.reshape(-1, self.inp_caps * self.nhidden)))
        elif hasattr(self, 'fc2'):
            self_z = F.relu(self.fc2(self_vectors.reshape(-1, self.inp_caps * self.nhidden)))
            neighbor_z = F.relu(self.fc2(neighbor_vectors.reshape(-1, self.inp_caps * self.nhidden)))
        else:
            self_z = self_vectors.reshape(-1, self.k * self.nhidden)
            neighbor_z = neighbor_vectors.reshape(-1, self.k * self.nhidden)

        if self.edge_feature_dim > 0:
            if not self.lora_edge_feats:
                edge_feats = F.relu(self.edge_fc(edge_feats.reshape(-1, self.edge_feature_dim)))
                neighbor_z = neighbor_z + edge_feats

        self_z_n = F.normalize(self_z.reshape(batch_size, num_considered_nodes, self.k, self.nhidden), dim=-1)
        neighbor_z_n = F.normalize(
            neighbor_z.reshape(batch_size, num_considered_nodes, n_neighbors, self.k, self.nhidden), dim=-1)

        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = torch.zeros(batch_size, num_considered_nodes, n_neighbors, self.k).to(
                    self_z_n.device)
            else:
                p = torch.sum(neighbor_z_n * u.view(batch_size, num_considered_nodes, 1, self.k, self.nhidden), dim=-1)
            p = F.softmax(p / self.tau, dim=-1)

            u = torch.sum(neighbor_z_n * p.view(batch_size, num_considered_nodes, n_neighbors, self.k, 1), dim=2)
            u += self_z_n
            if clus_iter < max_iter - 1:
                u = F.normalize(u, dim=-1)

        return self.drop(F.relu(u.reshape(batch_size, num_considered_nodes, self.k * self.nhidden)))

