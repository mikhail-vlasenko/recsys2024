import numpy as np
import json
import random
import time
import datetime
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from scipy import sparse
from collections import defaultdict

from tqdm import tqdm

from src.data.data_loader import random_neighbor, optimized_random_neighbor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(args, model, train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news,
                test_news_user, news_title, news_entity, news_group):
    model.to(device)

    train_data = np.array(train_data[:, [0, 1, 3]], dtype=np.int32)

    train_dataset = TensorDataset(
        torch.tensor(train_data[:, 0], dtype=torch.long),
        torch.tensor(train_data[:, 1], dtype=torch.long),
        torch.tensor(train_data[:, 2], dtype=torch.float32),
        # torch.tensor(train_user_news, dtype=torch.long),
        # torch.tensor(train_news_user, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.optimized_subsampling:
        # dict to list
        max_news_id = len(news_title)
        temp_train_news_user = []
        for i in range(max_news_id):
            if i in train_news_user:
                temp_train_news_user.append(train_news_user[i])
            else:
                temp_train_news_user.append([])
        train_news_user = temp_train_news_user
        user_lengths = torch.tensor([len(train_user_news[i]) for i in range(len(train_user_news))]).unsqueeze(1)#.to(device)
        news_lengths = torch.tensor([len(train_news_user[i]) for i in range(len(train_news_user))]).unsqueeze(1)#.to(device)

        def list_of_lists_to_torch(lst, pad_value, max_len, device):
            tensors = []
            for l in tqdm(
                    lst,
                    desc=f"Converting to torch (needs {max_len * len(lst) * 4 / 1024 / 1024:.2f} MB on {device})"
            ):
                pad = (0, max_len - len(l))  # pad 0 on the left and to max_len on the right
                tensors.append(F.pad(torch.tensor(l, dtype=torch.int32, device=device), pad, value=pad_value))
            return torch.stack(tensors)

        dense_matrix_device = torch.device("cpu")
        train_user_news = list_of_lists_to_torch(train_user_news, 0, user_lengths.max().item(), dense_matrix_device)
        train_news_user = list_of_lists_to_torch(train_news_user, 0, news_lengths.max().item(), dense_matrix_device)

    # eval_dataset = TensorDataset(
    #     torch.tensor(eval_data[:, 0], dtype=torch.long),
    #     torch.tensor(eval_data[:, 1], dtype=torch.long),
    #     torch.tensor(eval_data[:, 2], dtype=torch.float32),
    #     torch.tensor(test_user_news, dtype=torch.long),
    #     torch.tensor(test_news_user, dtype=torch.long)
    # )
    # eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        for user_indices, news_indices, labels in tqdm(train_loader):
            if args.optimized_subsampling:
                user_news, news_user = optimized_random_neighbor(args, train_user_news, train_news_user,
                                                                 user_lengths, news_lengths)
            else:
                user_news, news_user = random_neighbor(args, train_user_news, train_news_user, len(news_title))

            user_indices, news_indices, labels = user_indices.to(device), news_indices.to(device), labels.to(device)
            user_news, news_user = torch.tensor(user_news, dtype=torch.long).to(device), torch.tensor(
                news_user, dtype=torch.long).to(device)

            optimizer.zero_grad()
            scores, scores_normalized, predict_label, user_embeddings, news_embeddings = model(
                user_indices, news_indices, user_news, news_user
            )

            total_loss = criterion(scores, labels)
            infer_loss = model.infer_loss(user_embeddings, news_embeddings)
            loss = (1 - args.balance) * total_loss + args.balance * infer_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.n_epochs}, Loss: {total_loss / len(train_loader)}")

        # model.eval()
        # with torch.no_grad():
        #     eval_loss = 0
        #     all_labels = []
        #     all_scores = []
        #     for user_indices, news_indices, labels, user_news, news_user in eval_loader:
        #         user_indices, news_indices, labels = user_indices.to(device), news_indices.to(device), labels.to(device)
        #         user_news, news_user = user_news.to(device), news_user.to(device)
        #
        #         loss, scores_normalized, predict_label = model(user_indices, news_indices, labels, user_news, news_user)
        #         eval_loss += loss.item()
        #         all_labels.extend(labels.cpu().numpy())
        #         all_scores.extend(scores_normalized.cpu().numpy())
        #
        #     eval_auc = roc_auc_score(all_labels, all_scores)
        #     eval_f1 = f1_score(all_labels, [1 if score >= 0.5 else 0 for score in all_scores])
        #
        # print(
        #     f"Epoch {epoch + 1}/{args.n_epochs}, Eval Loss: {eval_loss / len(eval_loader)}, Eval AUC: {eval_auc}, Eval F1: {eval_f1}")

    return model
