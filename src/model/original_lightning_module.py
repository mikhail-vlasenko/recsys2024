from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, classification
from src.model.components.model import Model as Net
from src.data.data_loader import random_neighbor, optimized_random_neighbor
import logging
from tqdm import tqdm


class OriginalModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        train_user_news: list[list[float]],
        train_news_user: list[list[float]],
        n_news: int,
        args 
    ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.train_news_user = train_news_user
        self.train_user_news = train_user_news
        self.n_news = n_news

        self.net = net

        # loss function
        self.criterion = F.binary_cross_entropy_with_logits

        if args.optimized_subsampling:
            print("args.optimized_subsampling:", args.optimized_subsampling)
            self.train_user_news, self.train_news_user = self.pre_load_neighbors(train_user_news, train_news_user)
            #self.val_user_news, self.val_news_user = self.pre_load_neighbors(val_user_news, val_news_user)


        self.f1 = classification.BinaryF1Score()
        self.auc = classification.BinaryAUROC()

        self.more_labels = args.more_labels

        # make a set-based index for edges
        self.user_edge_index: list[set] = []
        for i in range(len(train_user_news)):
            # take the first column because it's the news index
            self.user_edge_index.append(set(np.array(train_user_news[i])[:, 0]))

    def pre_load_neighbors(self, user_news, news_user):
        user_lengths = torch.tensor([len(user_news[i]) for i in range(len(user_news))]).unsqueeze(1)#.to(device)
        news_lengths = torch.tensor([len(news_user[i]) for i in range(len(news_user))]).unsqueeze(1)#.to(device)
        self.user_lengths = user_lengths
        self.news_lengths = news_lengths

        def list_of_lists_to_torch(lst, pad_value, max_len, device):
            tensors = []
            for l in tqdm(
                    lst,
                    desc=f"Converting to torch (needs {max_len * len(lst) * 4 / 1024 / 1024:.2f} MB on {device})"
            ):
                pad = (0, max_len - len(l))  #pad 0 on the left and to max_len on the right
                tensors.append(F.pad(torch.tensor(l, dtype=torch.int32, device=device), pad, value=pad_value))
            return torch.stack(tensors)

        dense_matrix_device = torch.device("cpu")
        user_news = list_of_lists_to_torch(user_news, 0, user_lengths.max().item(), dense_matrix_device)
        news_user = list_of_lists_to_torch(news_user, 0, news_lengths.max().item(), dense_matrix_device)
        return user_news, news_user

    def load_batch(self, batch, mode="train"):
        user_id, article_index, labels = batch
        
        assert mode == "train"
        if mode == "train":
            user_news, news_user = self.train_user_news, self.train_news_user
        elif mode == "val":
            assert False, "Val mode not implemented"
        elif mode == "test":
            assert False, "Test mode not implemented"

        if self.hparams.args.optimized_subsampling:
            user_news, news_user = optimized_random_neighbor(self.hparams.args, user_news, news_user, self.user_lengths, self.news_lengths)
        else:
            user_news, news_user = random_neighbor(self.hparams.args, user_news, news_user, 2)

        user_news, news_user = torch.tensor(user_news, dtype=torch.float32).to(self.device), torch.tensor(
                news_user, dtype=torch.float32).to(self.device)
        
        return user_id, article_index, user_news, news_user, labels

    def compute_loss(self, scores, labels, user_embeddings, news_embeddings):
        total_loss = self.criterion(scores, labels.float())

        infer_loss = self.net.infer_loss(user_embeddings, news_embeddings)

        loss = (1 - self.hparams.args.balance) * total_loss + self.hparams.args.balance * infer_loss

        return loss

    def compute_scores(self, user_projected, news_projected, user_indices, news_indices, labels):
        if self.more_labels:
            # get label matrix using a list of sets for each user
            labels = torch.empty([len(user_indices), len(user_indices)], dtype=torch.float32)
            # converting to np is way faster than gpu_tensor.item()
            np_user_indices = user_indices.cpu().numpy()
            np_news_indices = news_indices.cpu().numpy()
            for i in range(len(np_user_indices)):
                for j in range(len(np_news_indices)):
                    labels[i, j] = 1 if np_news_indices[j] in self.user_edge_index[np_user_indices[i]] else 0
            labels = labels.flatten().to(user_projected.device)

            # matmul to get a matrix of similarities
            scores = torch.matmul(user_projected, news_projected.T)
            scores = scores.flatten()
        else:
            scores = self.net.get_edge_probability(user_projected, news_projected)

        return scores, labels

    def loss_from_batch(
            self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str, ret_scores=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        user_indices, news_indices, user_news, news_user, labels = self.load_batch(batch, mode=mode)

        user_embeddings, news_embeddings = self.net(user_indices, news_indices, user_news, news_user)
        user_projected, news_projected = self.net.apply_projection(user_embeddings, news_embeddings)
        scores, labels = self.compute_scores(user_projected, news_projected, user_indices, news_indices, labels)
        loss = self.compute_loss(scores, labels, user_embeddings, news_embeddings)

        if ret_scores:
            return loss, scores, labels
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss = self.loss_from_batch(batch, mode="train")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, scores, labels = self.loss_from_batch(batch, mode = "train", ret_scores=True) #TODO change mode to val

        f1 = self.f1(scores, labels)
        roc_auc = self.auc(scores, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1", f1, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/roc_auc", roc_auc, on_epoch=True, prog_bar=True, logger=True)

        return loss, f1, roc_auc

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        loss, scores, labels = self.loss_from_batch(batch, mode="test", ret_scores=True)

        f1 = self.f1(scores, labels)
        roc_auc = self.auc(scores, labels)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/f1", f1, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/roc_auc", roc_auc, on_epoch=True, prog_bar=True, logger=True)

        return loss, f1, roc_auc

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.hparams.args.lr, weight_decay=self.hparams.args.l2_weight
        )
        return {"optimizer": optimizer}

    def to(self, *args: Any, **kwargs: Any) -> 'OriginalModule':
        super().to(*args, **kwargs)
        # idk why but pl doesn't call it like that
        self.net.to(*args, **kwargs)
        return self
