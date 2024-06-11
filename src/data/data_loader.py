import numpy as np
import json


def load_new_data(args):
    r = np.load("./data/data.npz", allow_pickle=True)
    train_data = r['train_data']
    test_data = r['test_data']
    news_entity = r['news_entity']
    news_group = r['news_group']
    news_title = r['news_title'][:, :args.title_len]
    #I think above are precomputed emedddings and below are the actual data

    with open("./data/train_user_news.txt", 'r') as file:
        train_user_news = eval(file.read())
    print('train_user_news load over!')
    with open("./data/test_user_news.txt", 'r') as file:
        test_user_news = eval(file.read())
    print('test_user_news load over!')
    with open("./data/train_news_user.json", 'r') as file:
        train_news_user = json.load(file)
        train_news_user = dict(zip(list(map(int, train_news_user.keys())), train_news_user.values()))
    print('train_news_user load over!')
    with open("./data/test_news_user.json", 'r') as file:
        test_news_user = json.load(file)
        test_news_user = dict(zip(list(map(int, test_news_user.keys())), test_news_user.values()))
    print('test_news_user load over!')
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    l = int(len(test_data) * 0.1)
    eval_data = test_data[:l]
    test_data = test_data[l:]

    # train_data columns are user_indices, news_indices, some_bullshit, labels

    #args, torch.tensor(news_title), torch.tensor(news_entity), torch.tensor(news_group), len(train_user_news), len(news_title)

    return train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, \
        news_title, news_entity, news_group


def train_random_neighbor(args, train_user_news, train_news_user, news_len):
    user_news = np.zeros([len(train_user_news), args.news_neighbor], dtype=np.int32)
    for i in range(1, len(train_user_news)):
        n_neighbors = len(train_user_news[i])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=True)
        user_news[i] = np.array([train_user_news[i][k] for k in sampled_indices])

    news_user = np.zeros([news_len, args.user_neighbor], dtype=np.int32)
    for i in train_news_user:
        n_neighbors = len(train_news_user[i])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=True)
        news_user[int(i)] = np.array([train_news_user[i][k] for k in sampled_indices])

    return user_news, news_user


def test_random_neighbor(args, test_user_news, test_news_user, news_len):
    user_news = np.zeros([len(test_user_news), args.news_neighbor], dtype=np.int32)
    for i in range(1, len(test_user_news)):
        n_neighbors = len(test_user_news[i])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor, replace=True)
        user_news[i] = np.array([test_user_news[i][k] for k in sampled_indices])

    news_user = np.zeros([news_len, args.user_neighbor], dtype=np.int32)
    for i in test_news_user:
        n_neighbors = len(test_news_user[i])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor, replace=True)
        news_user[int(i)] = np.array([test_news_user[i][k] for k in sampled_indices])

    return user_news, news_user
