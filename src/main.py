import argparse
import pickle

import numpy as np
import time

import torch

from src.data.data_loader import load_new_data
from src.model.components.model import Model
from src.train import train_model
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='ten_week', help='which dataset to use')
parser.add_argument('--title_len', type=int, default=10, help='the max length of title')
parser.add_argument('--session_len', type=int, default=10, help='the max length of session')
parser.add_argument('--aggregator', type=str, default='neighbor', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--user_neighbor', type=int, default=30, help='the number of neighbors to be sampled')
parser.add_argument('--news_neighbor', type=int, default=10, help='the number of neighbors to be sampled')
parser.add_argument('--entity_neighbor', type=int, default=1, help='the number of neighbors to be sampled') #whats this one
parser.add_argument('--user_dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--cnn_out_size', type=int, default=128, help='dimension of cnn output')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=5e-3, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  #3e-4
parser.add_argument('--save_path', type=str, default="./data/1week/hop2/version1/", help='model save path')
parser.add_argument('--test', type=int, default=0, help='test')
parser.add_argument('--use_group', type=int, default=1, help='whether use group')
parser.add_argument('--n_filters', type=int, default=64, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[2, 3], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')
parser.add_argument('--ncaps', type=int, default=7,
                    help='Maximum number of capsules per layer.')
parser.add_argument('--dcaps', type=int, default=0, help='Decrease this number of capsules per layer.')
parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
parser.add_argument('--routit', type=int, default=7,
                    help='Number of iterations when routing.')
parser.add_argument('--balance', type=float, default=0.004, help='learning rate')  #3e-4
parser.add_argument('--version', type=int, default=0,
                        help='Different version under the same set')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
parser.add_argument('--optimized_subsampling', type=bool, default=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

show_loss = True
show_time = False

args = parser.parse_args()

fast_load_path = 'data.pkl'

if not os.path.exists(fast_load_path):
    data_tuple = load_new_data(args)

    with open(fast_load_path, 'wb') as f:
        pickle.dump(data_tuple, f)

with open(fast_load_path, 'rb') as f:
    data_tuple = pickle.load(f)

train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, news_title, news_entity, news_group = data_tuple

#len(train_user_news) -> is n users  

model = Model(
    args,
    torch.tensor(news_title).to(device), torch.tensor(news_entity).to(device), torch.tensor(news_group).to(device),
    len(train_user_news), len(news_title)
)

trained_model = train_model(args, model, train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, news_title, news_entity, news_group)
