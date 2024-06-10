from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from src.model.components.model import Model as Net
import logging


class Wang2022LightningModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
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

        self.net = net

        # loss function
        self.criterion = torch.nn.binary_cross_entropy_with_logits()


    def forward(self, user_indices, news_indices, user_news, news_user, labels) -> torch.Tensor:

        out = self.net(user_indices, news_indices, user_news, news_user, labels)
        
        return out

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
        print(batch)
        user_indices, news_indices, user_news, news_user, labels = batch

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
        user_indices, news_indices, user_news, news_user, labels = batch

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
        user_indices, news_indices, user_news, news_user, labels = batch

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
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}