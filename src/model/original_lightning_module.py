from typing import Any, Dict, Tuple, Union, Optional
from typing_extensions import Self

import torch
import torch.nn.functional as F
from lightning import LightningModule
from src.model.components.model import Model 
from src.data.data_loader import random_neighbor, optimized_random_neighbor
from src.ebrec.evaluation.metrics_protocols import MetricEvaluator, AucScore, MrrScore, NdcgScore
import logging
from tqdm import tqdm


class OriginalModule(LightningModule):
    def __init__(
        self,
        net: Model,
        train_user_news: list[list[int]],
        train_news_user: list[list[int]],
        val_user_news: list[list[int]],
        val_news_user: list[list[int]],
        train_article_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        val_article_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        n_users: int,
        args 
    ) -> None:
        
        super(OriginalModule,self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.train_news_user = train_news_user
        self.train_user_news = train_user_news
        self.val_news_user = val_news_user
        self.val_user_news = val_user_news
        
        #set up article features
        self.train_news_title, self.train_news_entity, self.train_news_group = train_article_features
        self.val_news_title, self.val_news_entity, self.val_news_group = val_article_features
        self.train_news_title, self.train_news_entity, self.train_news_group = self.train_news_title.to(self.device), self.train_news_entity.to(self.device), self.train_news_group.to(self.device)
        self.val_news_title, self.val_news_entity, self.val_news_group = self.val_news_title.to(self.device), self.val_news_entity.to(self.device), self.val_news_group.to(self.device)
        
        self.net = net

        # loss function
        self.criterion = F.binary_cross_entropy_with_logits

        if args.optimized_subsampling:
            print("args.optimized_subsampling:", args.optimized_subsampling)
            self.train_user_news, self.train_news_user = self.pre_load_neighbors(train_user_news, train_news_user)
            self.val_user_news, self.val_news_user = self.pre_load_neighbors(val_user_news, val_news_user)


        self.metrics = MetricEvaluator(labels=[], predictions=[], predictions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)])

        self.more_labels = args.more_labels

        # make a set-based index for edges
        self.train_user_edge_index: list[set] = []
        for i in range(len(train_user_news)):
            self.train_user_edge_index.append(set(train_user_news[i]))

        self.val_user_edge_index: list[set] = []
        for i in range(len(val_user_news)):
            self.val_user_edge_index.append(set(val_user_news[i]))
        
        #TODO add test set

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
        #mode='train'
        #assert mode == "train"
        if mode == "train":
            user_news, news_user = self.train_user_news, self.train_news_user
        elif mode == "val":
            user_news, news_user = self.val_user_news, self.val_news_user
        elif mode == "test":
            assert False, "Test mode not implemented"

        if self.hparams.args.optimized_subsampling:
            user_news, news_user = optimized_random_neighbor(self.hparams.args, user_news, news_user, self.user_lengths, self.news_lengths)
        else:
            user_news, news_user = random_neighbor(self.hparams.args, user_news, news_user)

        user_news, news_user = torch.tensor(user_news, dtype=torch.long).to(self.device), torch.tensor(
                news_user, dtype=torch.long).to(self.device)
        
        return user_id, article_index, user_news, news_user, labels

    def compute_loss(self, scores, labels, user_embeddings, news_embeddings):
        total_loss = self.criterion(scores, labels.float())

        infer_loss = self.net.infer_loss(user_embeddings, news_embeddings)

        loss = (1 - self.hparams.args.balance) * total_loss + self.hparams.args.balance * infer_loss

        return loss

    def compute_scores(self, user_projected, news_projected, user_indices, news_indices, labels, mode):
        if self.more_labels:
            if mode == "train":
                user_edge_index = self.train_user_edge_index
            elif mode == "val":
                user_edge_index = self.val_user_edge_index
            else:
                assert False, "Test mode not implemented"
            # get label matrix using a list of sets for each user
            labels = torch.empty([len(user_indices), len(user_indices)], dtype=torch.float32)
            for i in range(len(user_indices)):
                for j in range(len(news_indices)):
                    labels[i, j] = 1 if j in user_edge_index[i] else 0
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
        user_indices, news_indices, user_news, news_user, labels = self.load_batch(batch, mode = mode)

        user_embeddings, news_embeddings = self.net(user_indices, news_indices, user_news, news_user)
        user_projected, news_projected = self.net.apply_projection(user_embeddings, news_embeddings)
        scores, labels = self.compute_scores(user_projected, news_projected, user_indices, news_indices, labels, mode=mode)
        loss = self.compute_loss(scores, labels, user_embeddings, news_embeddings)

        if ret_scores:
            return loss, scores, labels
        return loss
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins.
        We need to set the model to training mode here. -> swap out the article features to the train ones 
        """  
        self.net.train()
        self.metrics = MetricEvaluator(labels=[], predictions=[], predictions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)])
        train_news_title, train_news_entity, train_news_group = self.train_news_title.to(self.device), self.train_news_entity.to(self.device), self.train_news_group.to(self.device)
        self.net.set_article_features(train_news_title, train_news_entity, train_news_group)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss, scores, labels = self.loss_from_batch(batch, mode="train", ret_scores=True)

        self.metrics.labels += [labels.cpu().numpy()]
        self.metrics.predictions += [scores.cpu().numpy()]
        metric_dict = self.metrics.evaluate()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/ndcg@10", metric_dict['ndcg@10'], on_epoch=True, prog_bar=True, logger=True)
        self.log("train/auc", metric_dict['auc'], on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_validation_start(self) -> None:
        """Lightning hook that is called when validation begins.
        We need to set the model to evaluation mode here. -> swap out the article features to the val ones
        """
        self.net.eval()
        self.metrics = MetricEvaluator(labels=[], predictions=[], predictions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)])

        val_news_title, val_news_entity, val_news_group = self.val_news_title.to(self.device), self.val_news_entity.to(self.device), self.val_news_group.to(self.device)
        self.net.set_article_features(val_news_title, val_news_entity, val_news_group)


    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, scores, labels = self.loss_from_batch(batch, mode = "val", ret_scores=True) #TODO change mode to val

        self.metrics.labels += [labels.cpu().numpy()]
        self.metrics.predictions += [scores.cpu().numpy()]
        metric_dict = self.metrics.evaluate() #gives a rolling computation of the metrics

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/ndcg@10", metric_dict['ndcg@10'], on_epoch=True, prog_bar=True, logger=True)
        self.log("val/auc", metric_dict['auc'], on_epoch=True, prog_bar=True, logger=True)
        self.log("val/mrr", metric_dict['mrr'], on_epoch=True, prog_bar=False, logger=True)
        self.log("val/ndcg@5", metric_dict['ndcg@5'], on_epoch=True, prog_bar=False, logger=True)
        #also log the beyond accuracy metrics to the logger
        
        return loss, metric_dict['ndcg@10'], metric_dict['auc']
    
    def on_test_start(self) -> None:
        """Lightning hook that is called when test begins.
        We need to set the model to evaluation mode here. -> swap out the article features to the test ones
        """
        self.net.eval()
        self.metrics = MetricEvaluator(labels=[], predictions=[], predictions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)])

        test_news_title, test_news_entity, test_news_group = self.test_news_title.to(self.device), self.test_news_entity.to(self.device), self.test_news_group.to(self.device)
        self.net.set_article_features(test_news_title, test_news_entity, test_news_group)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        
        loss, scores, labels = self.loss_from_batch(batch, mode="test", ret_scores=True)
        
        self.metrics.labels += [labels.cpu().numpy()]
        self.metrics.predictions += [scores.cpu().numpy()]
        metric_dict = self.metrics.evaluate() #gives a rolling computation of the metrics

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/f1", metric_dict['ndcg@10'], on_epoch=True, prog_bar=True, logger=True)
        self.log("test/auc", metric_dict['auc'], on_epoch=True, prog_bar=True, logger=True)
        self.log("test/mrr", metric_dict['mrr'], on_epoch=True, prog_bar=False, logger=True)
        self.log("test/ndcg@5", metric_dict['ndcg@5'], on_epoch=True, prog_bar=False, logger=True)
        #also log the beyond accuracy metrics to the logger
        
        return loss, metric_dict['ndcg@10'], metric_dict['auc']

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.hparams.args.lr, weight_decay=self.hparams.args.l2_weight
        )
        return {"optimizer": optimizer}

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        # idk why but pl doesn't call it like that
        self.net.to(*args, **kwargs)
        return self
