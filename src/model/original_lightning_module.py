from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.model.components.model import Model as Net
from src.data.data_loader import random_neighbor, optimized_random_neighbor
import logging
from tqdm import tqdm


class OriginalModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        compile: bool,
        train_user_news,
        train_news_user,
        n_news,
        id_to_index,
        args: bool
    ) -> None:
        """Initialize a `pl_classifier`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.train_news_user = train_news_user
        self.train_user_news = train_user_news
        self.n_news = n_news
        self.id_to_index = id_to_index

        self.net = net

        # loss function
        self.criterion = F.binary_cross_entropy_with_logits

        if not args.optimized_subsampling:
            print(args.optimized_subsampling)
            self.pre_load_neighbors()
        

    def pre_load_neighbors(self):
        max_news_id = self.n_news
        temp_train_news_user = []
        for i in range(max_news_id):
            if i in self.train_news_user:
                temp_train_news_user.append(train_news_user[i])
            else:
                temp_train_news_user.append([])
        train_news_user = temp_train_news_user
        user_lengths = torch.tensor([len(self.train_user_news[i]) for i in range(len(self.train_user_news))]).unsqueeze(1)#.to(device)
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
        self.train_user_news = list_of_lists_to_torch(self.train_user_news, 0, user_lengths.max().item(), dense_matrix_device)
        self.train_news_user = list_of_lists_to_torch(self.train_news_user, 0, news_lengths.max().item(), dense_matrix_device)


    def load_batch(self, batch, mode="train"):
        user_id, article_id, labels = batch

        article_index = self.id_to_index[article_id].to(article_id.device)
        
        if mode == "train":
            user_news, news_user = self.train_user_news, self.train_news_user
        
        user_news, news_user = torch.tensor(user_news, dtype=torch.long).to(user_id.device), torch.tensor(
                news_user, dtype=torch.long).to(user_id.device)
        
        return user_id, article_index, user_news, news_user, labels

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def compute_loss(self, scores, labels, user_embeddings, news_embeddings):
        total_loss = self.criterion(scores, labels)

        l2_loss = sum(torch.norm(param) for param in self.parameters())
        infer_loss, ret_w = self.net.infer_loss(user_embeddings, news_embeddings)

        loss = (1 - self.balance) * total_loss + self.balance * infer_loss + self.l2_weight * l2_loss

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
        
        user_indices, news_indices, user_news, news_user, labels = self.load_batch(batch)

        scores, user_embeddings, news_embeddings = self.net(user_indices, news_indices, user_news, news_user, labels)

        loss = self.compute_loss(scores, labels, user_embeddings, news_embeddings)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        user_indices, news_indices, user_news, news_user, labels = self.load_batch(batch)

        scores, user_embeddings, news_embeddings = self.net(user_indices, news_indices, user_news, news_user, labels)

        loss = self.compute_loss(scores, labels, user_embeddings, news_embeddings)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        user_indices, news_indices, user_news, news_user, labels = self.load_batch(batch)

        scores, user_embeddings, news_embeddings = self.net(user_indices, news_indices, user_news, news_user, labels)

        loss = self.compute_loss(scores, labels, user_embeddings, news_embeddings)

        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.args.lr, 
                                      weight_decay=self.hparams.args.l2_weight)
        return {"optimizer": optimizer}