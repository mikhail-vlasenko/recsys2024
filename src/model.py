import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from src.aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, RoutingLayer


class Model(nn.Module):
    def __init__(self, args, news_title, news_entity, news_group, n_user, n_news):
        super(Model, self).__init__()

        n_word = 279215
        self.use_group = args.use_group
        self.n_filters = args.n_filters
        self.filter_sizes = args.filter_sizes
        self.max_session_len = args.session_len
        self.user_dim = args.user_dim
        self.lr = args.lr
        self.title_len = args.title_len
        self.batch_size = args.batch_size
        self.news_neighbor = args.news_neighbor
        self.user_neighbor = args.user_neighbor
        self.entity_neighbor = args.entity_neighbor
        self.n_iter = args.n_iter
        self.l2_weight = args.l2_weight
        self.cnn_out_size = args.cnn_out_size

        self.news_entity = news_entity
        self.news_group = news_group

        self.title = news_title
        self.ncaps = args.ncaps
        self.dcaps = args.dcaps
        self.nhidden = args.nhidden
        self.dim = self.ncaps * self.nhidden
        self.routit = args.routit
        self.balance = args.balance

        self.n_user = n_user
        self.n_news = n_news

        self.group_embedding = nn.Parameter(torch.randn(12, 50) * 0.1)
        self.user_emb_matrix = nn.Parameter(torch.randn(n_user + 1, self.user_dim) * 0.1)
        self.word_emb_matrix = nn.Parameter(torch.randn(n_word + 1, 50) * 0.1)

        self.filter_shape_item = [40, 20, 1, 8]
        self.input_size_item = 10 * 8 * 8
        self.filter_shape_title = [2, 20, 1, 8]
        self.input_size_title = 4 * 8 * 8
        self.filter_shape = [2, 8, 1, 4]
        self.cat_size = 7 * 30 * 4

        self.Routing = RoutingLayer

        self.dropout_rate = nn.Dropout(p=args.dropout_rate)
        self.conv_layers = nn.ModuleDict()
        self.build_model()

    def build_model(self):
        # self.user_emb_matrix = F.normalize(self.user_emb_matrix, dim=-1)
        # self.word_emb_matrix = F.normalize(self.word_emb_matrix, dim=-1)

        self.conv_layers['item'] = nn.Conv2d(1, 8, kernel_size=(40, 20), stride=(2, 2))
        self.conv_layers['title'] = nn.Conv2d(1, 8, kernel_size=(2, 20), stride=(2, 2))

        self.pool_item = nn.MaxPool2d(kernel_size=(3, 2), stride=(2, 2))
        self.pool_title = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 2))

        self.dense1 = nn.Linear(self.input_size_item + self.input_size_title, self.cnn_out_size)
        # todo: where do 10 and 30 come from?
        self.dense2 = nn.Linear(self.dense1.in_features * 10, self.cnn_out_size)
        self.dense3 = nn.Linear(self.dense2.in_features * 30, self.cnn_out_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, user_indices, news_indices, user_news, news_user, labels):
        newsvec, uservec = self.get_neighbors(news_indices, user_indices, user_news, news_user)
        news_embeddings, user_embeddings, aggregators = self.aggregate(newsvec, uservec)

        scores = torch.squeeze(self.simple_dot_net(user_embeddings, news_embeddings))
        scores_normalized = torch.sigmoid(scores)
        predict_label = (scores > 0.5).int()

        total_loss = F.binary_cross_entropy_with_logits(scores, labels)

        l2_loss = sum(torch.norm(param) for param in self.parameters())
        infer_loss, ret_w = self.infer_loss(user_embeddings, news_embeddings)

        loss = (1 - self.balance) * total_loss + self.balance * infer_loss + self.l2_weight * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, scores_normalized, predict_label

    def simple_dot_net(self, x, y):
        caps = self.ncaps - (self.n_iter - 1) * self.dcaps
        last_w = nn.Parameter(torch.randn(caps * self.nhidden, caps * self.nhidden) * 0.1)
        last_b = nn.Parameter(torch.zeros(caps * self.nhidden))

        x_map = torch.matmul(x[-1].reshape(self.batch_size, -1), last_w) + last_b
        y_map = torch.matmul(y[-1].reshape(self.batch_size, -1), last_w) + last_b

        output = torch.sum(x_map * y_map, dim=-1)
        return output

    def infer_loss(self, x, y):
        caps = self.ncaps - (self.n_iter - 1) * self.dcaps
        ret_uw = nn.Parameter(torch.randn(self.nhidden, caps) * 0.1)
        ret_ub = nn.Parameter(torch.zeros(caps))

        x_class = torch.matmul(x[-1].reshape(-1, self.nhidden), ret_uw) + ret_ub
        y_class = torch.matmul(y[-1].reshape(-1, self.nhidden), ret_uw) + ret_ub

        label = torch.eye(caps).repeat(self.batch_size, 1)
        user_infer_loss = torch.mean(torch.sum(F.cross_entropy(x_class, label)))
        news_infer_loss = torch.mean(torch.sum(F.cross_entropy(y_class, label)))

        loss = user_infer_loss + news_infer_loss

        return loss, ret_uw

    def get_neighbors(self, news_seeds, user_seeds, user_news, news_user):
        news_seeds = news_seeds.unsqueeze(1)
        user_seeds = user_seeds.unsqueeze(1)
        news = [news_seeds]
        user = [user_seeds]
        news_vectors = []
        user_vectors = []
        n = self.news_neighbor
        u = self.user_neighbor

        user_weights = nn.Parameter(torch.randn(self.user_dim, self.dim) * 0.1)
        user_bias = nn.Parameter(torch.randn(self.dim) * 0.1)
        item_weights = nn.Parameter(torch.randn(self.cnn_out_size, self.dim) * 0.1)
        item_bias = nn.Parameter(torch.randn(self.dim) * 0.1)

        news_hop_vectors = self.convolution(news[0]).reshape(-1, self.cnn_out_size)
        news_hop_vectors = F.relu(torch.matmul(news_hop_vectors, item_weights) + item_bias)
        news_vectors.append(news_hop_vectors.reshape(self.batch_size, -1, self.dim))
        news_neighbors = F.embedding(news[0][:, 0], news_user)
        news.append(news_neighbors)

        user_hop_vectors = F.embedding(user[0], self.user_emb_matrix).reshape(-1, self.user_dim)
        user_hop_vectors = F.relu(torch.matmul(user_hop_vectors, user_weights) + user_bias)
        user_vectors.append(user_hop_vectors.reshape(self.batch_size, -1, self.dim))
        user_neighbors = F.embedding(user[0][:, 0], user_news)
        user.append(user_neighbors)

        if self.n_iter >= 1:
            news_hop_vectors = F.embedding(news[1][:, :u], self.user_emb_matrix).reshape(-1, self.user_dim)
            news_hop_vectors = F.relu(torch.matmul(news_hop_vectors, user_weights) + user_bias)
            news_hop_vectors = news_hop_vectors.reshape(self.batch_size, -1, self.dim)
            news_neighbors = user_news[news[1][:, :u]].view(self.batch_size, -1)
            news_vectors.append(news_hop_vectors)
            news.append(news_neighbors)

            user_hop_vectors = self.convolution(user[1]).reshape(-1, self.cnn_out_size)
            user_hop_vectors = F.relu(torch.matmul(user_hop_vectors, item_weights) + item_bias)
            user_hop_vectors = user_hop_vectors.reshape(self.batch_size, -1, self.dim)
            user_neighbors = news_user[user[1][:, :n]].view(self.batch_size, -1)
            user_vectors.append(user_hop_vectors)
            user.append(user_neighbors)

        if self.n_iter >= 2:
            news_hop_vectors = self.convolution(news[2]).reshape(-1, self.cnn_out_size)
            news_hop_vectors = F.relu(torch.matmul(news_hop_vectors, item_weights) + item_bias)
            news_hop_vectors = news_hop_vectors.reshape(self.batch_size, -1, self.dim)
            news_neighbors = news_user[news[2]].view(self.batch_size, -1)
            news_vectors.append(news_hop_vectors)
            news.append(news_neighbors)

            user_hop_vectors = F.embedding(user[2], self.user_emb_matrix).reshape(-1, self.user_dim)
            user_hop_vectors = F.relu(torch.matmul(user_hop_vectors, user_weights) + user_bias)
            user_hop_vectors = user_hop_vectors.reshape(self.batch_size, -1, self.dim)
            user_neighbors = user_news[user[2]].view(self.batch_size, -1)
            user_vectors.append(user_hop_vectors)
            user.append(user_neighbors)

        # n_iter is always 2
        return news_vectors, user_vectors

    def aggregate(self, news_vectors, user_vectors):
        conv_ls = []
        conv = None
        inp_caps, out_caps = None, self.ncaps
        cur_dim = self.dim

        news = []
        user = []
        for i in range(self.n_iter):
            conv = self.Routing(i, out_caps, self.nhidden, self.batch_size, self.dropout_rate, inp_caps)

            conv_ls.append(conv)

            news_vectors_next_iter = []
            user_vectors_next_iter = []

            for hop in range(self.n_iter - i):
                if hop % 2 == 0:
                    if inp_caps is None:
                        news_shape = (self.batch_size, -1, self.user_neighbor, self.dim)
                        user_shape = (self.batch_size, -1, self.news_neighbor, self.dim)
                    else:
                        news_shape = (self.batch_size, -1, self.user_neighbor, inp_caps * self.nhidden)
                        user_shape = (self.batch_size, -1, self.news_neighbor, inp_caps * self.nhidden)
                else:
                    if inp_caps is None:
                        news_shape = (self.batch_size, -1, self.news_neighbor, self.dim)
                        user_shape = (self.batch_size, -1, self.user_neighbor, self.dim)
                    else:
                        news_shape = (self.batch_size, -1, self.news_neighbor, inp_caps * self.nhidden)
                        user_shape = (self.batch_size, -1, self.user_neighbor, inp_caps * self.nhidden)

                news_vector = conv.rout(self_vectors=news_vectors[hop],
                                        neighbor_vectors=news_vectors[hop + 1].reshape(*news_shape),
                                        max_iter=self.routit)
                user_vector = conv.rout(self_vectors=user_vectors[hop],
                                        neighbor_vectors=user_vectors[hop + 1].reshape(*user_shape),
                                        max_iter=self.routit)

                news_vectors_next_iter.append(news_vector)
                user_vectors_next_iter.append(user_vector)

            news_vectors = news_vectors_next_iter
            user_vectors = user_vectors_next_iter

            news.append(news_vectors[0].reshape(self.batch_size, out_caps, self.nhidden))
            user.append(user_vectors[0].reshape(self.batch_size, out_caps, self.nhidden))

            cur_dim += out_caps * self.nhidden
            inp_caps, out_caps = out_caps, max(1, out_caps - self.dcaps)

        return news, user, conv_ls

    def convolution(self, inputs):
        title_lookup = F.embedding(inputs, self.title).reshape(-1, self.title_len)
        title_embed = F.embedding(title_lookup, self.word_emb_matrix).unsqueeze(-1)

        item_lookup = F.embedding(inputs, self.news_entity).reshape(-1, 40)
        group_lookup = F.embedding(inputs, self.news_group).reshape(-1, 40)
        item_embed = F.embedding(item_lookup, self.word_emb_matrix).unsqueeze(2)
        group_embed = F.embedding(group_lookup, self.group_embedding).unsqueeze(2)
        item_group_embed = torch.cat((item_embed, group_embed), 2).reshape(-1, 80, 50).unsqueeze(-1)

        # in tf1 conv input is NHWC, not NCHW
        print(item_group_embed.shape)
        item_group_embed = item_group_embed.permute(0, 3, 1, 2)
        conv_item = self.conv_layers['item'](item_group_embed)
        h_item = F.relu(conv_item)

        pooled_item = self.pool_item(h_item)
        # and put it back to NHWC for consistency
        pooled_item = pooled_item.permute(0, 2, 3, 1)
        pool_item = pooled_item.reshape(self.batch_size, -1, self.input_size_item)

        title_embed = title_embed.permute(0, 3, 1, 2)
        conv_title = self.conv_layers['title'](title_embed)
        h_title = F.relu(conv_title)

        pooled_title = self.pool_title(h_title)
        pooled_title = pooled_title.permute(0, 2, 3, 1)
        pool_title = pooled_title.reshape(self.batch_size, -1, self.input_size_title)

        pooled = torch.cat((pool_item, pool_title), -1)
        pooled = pooled.reshape(self.batch_size, -1)

        # convolution gets called with differnet input sizes so we actually need 3 dense layers and need to choose the right one
        # the original code makes 3 layers implicitly because its tensorflow
        if pooled.shape[1] == self.dense1.in_features:
            pool = self.dense1(pooled)
        elif pooled.shape[1] == self.dense2.in_features:
            pool = self.dense2(pooled)
        else:
            pool = self.dense3(pooled)
        # we need to flatten pooled and add a nonlinearity around the output. Relu in their case.
        pool = F.relu(pool)

        return pool

    # def train(self, batch_data):
    #     user_indices, news_indices, labels, user_news, news_user = batch_data
    #     loss, scores_normalized, predict_label = self(user_indices, news_indices, labels, user_news, news_user)
    #
    #     auc = roc_auc_score(labels.cpu().numpy(), scores_normalized.detach().cpu().numpy())
    #     f1 = f1_score(labels.cpu().numpy(), predict_label.cpu().numpy())
    #
    #     return loss, auc, f1
    #
    # def eval(self, batch_data):
    #     user_indices, news_indices, labels, user_news, news_user = batch_data
    #     _, scores_normalized, predict_label = self(user_indices, news_indices, labels, user_news, news_user)
    #
    #     auc = roc_auc_score(labels.cpu().numpy(), scores_normalized.detach().cpu().numpy())
    #     f1 = f1_score(labels.cpu().numpy(), predict_label.cpu().numpy())
    #
    #     return auc, f1
