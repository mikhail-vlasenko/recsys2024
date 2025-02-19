# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np

from src.ebrec.models.newsrec.base_model import BaseModel
from src.ebrec.models.newsrec.layers import AttLayer2

__all__ = ["NAMLModel"]


class NAMLModel(BaseModel):
    """NAML model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie,
    Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        n_users: int = 50000,
        word2vec_embedding=None,
        seed=None,
        **kwargs,
    ):
        """Initialization steps for NAML.
        Compared with the BaseModel, NAML need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train (object): NAML data loader class for train data.
            iterator_creator_test (object): NAML data loader class for test and validation data
        """

        self.n_users = n_users

        super().__init__(
            hparams=hparams,
            word2vec_embedding=word2vec_embedding,
            seed=seed,
            **kwargs,
        )

    def _build_graph(self):
        """Build NAML model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_naml()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NAML.

        Args:
            newsencoder (object): the news encoder of NAML.

        Return:
            object: the user encoder of NAML.
        """
        his_input_title_body_verts = keras.Input(
            shape=(
                self.hparams.history_size,
                self.hparams.title_size + self.hparams.body_size + 2,
            ),
            dtype="int32",
        )

        click_news_presents = layers.TimeDistributed(newsencoder)(
            his_input_title_body_verts
        )
        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(
            click_news_presents
        )

        model = keras.Model(
            his_input_title_body_verts, user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NAML.
        news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NAML.
        """
        input_title_body_verts = keras.Input(
            shape=(self.hparams.title_size + self.hparams.body_size + 2,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(
            lambda x: x[:, : self.hparams.title_size]
        )(input_title_body_verts)
        sequences_input_body = layers.Lambda(
            lambda x: x[
                :,
                self.hparams.title_size : self.hparams.title_size
                + self.hparams.body_size,
            ]
        )(input_title_body_verts)
        input_vert = layers.Lambda(
            lambda x: x[
                :,
                self.hparams.title_size
                + self.hparams.body_size : self.hparams.title_size
                + self.hparams.body_size
                + 1,
            ]
        )(input_title_body_verts)
        input_subvert = layers.Lambda(
            lambda x: x[:, self.hparams.title_size + self.hparams.body_size + 1 :]
        )(input_title_body_verts)

        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        vert_repr = self._build_vertencoder()(input_vert)
        subvert_repr = self._build_subvertencoder()(input_subvert)

        concate_repr = layers.Concatenate(axis=-2)(
            [title_repr, body_repr, vert_repr, subvert_repr]
        )
        news_repr = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(
            concate_repr
        )

        model = keras.Model(input_title_body_verts, news_repr, name="news_encoder")
        return model

    def _build_titleencoder(self, embedding_layer):
        """build title encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the title encoder of NAML.
        """

        sequences_input_title = keras.Input(
            shape=(self.hparams.title_size,), dtype="int32"
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(self.hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            self.hparams.filter_num,
            self.hparams.window_size,
            activation=self.hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(self.hparams.dropout)(y)
        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, self.hparams.filter_num))(pred_title)

        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        return model

    def _build_bodyencoder(self, embedding_layer):
        """build body encoder of NAML news encoder.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the body encoder of NAML.
        """

        sequences_input_body = keras.Input(
            shape=(self.hparams.body_size,), dtype="int32"
        )
        embedded_sequences_body = embedding_layer(sequences_input_body)

        y = layers.Dropout(self.hparams.dropout)(embedded_sequences_body)
        y = layers.Conv1D(
            self.hparams.filter_num,
            self.hparams.window_size,
            activation=self.hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(self.hparams.dropout)(y)
        pred_body = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_body = layers.Reshape((1, self.hparams.filter_num))(pred_body)

        model = keras.Model(sequences_input_body, pred_body, name="body_encoder")
        return model

    def _build_vertencoder(self):
        """build vert encoder of NAML news encoder.

        Return:
            object: the vert encoder of NAML.
        """
        input_vert = keras.Input(shape=(1,), dtype="int32")

        vert_embedding = layers.Embedding(
            self.hparams.vert_num, self.hparams.vert_emb_dim, trainable=True
        )

        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            self.hparams.filter_num,
            activation=self.hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(vert_emb)
        pred_vert = layers.Reshape((1, self.hparams.filter_num))(pred_vert)

        model = keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model

    def _build_subvertencoder(self):
        """build subvert encoder of NAML news encoder.

        Return:
            object: the subvert encoder of NAML.
        """

        input_subvert = keras.Input(shape=(1,), dtype="int32")

        subvert_embedding = layers.Embedding(
            self.hparams.subvert_num, self.hparams.subvert_emb_dim, trainable=True
        )

        subvert_emb = subvert_embedding(input_subvert)
        pred_subvert = layers.Dense(
            self.hparams.filter_num,
            activation=self.hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(subvert_emb)
        pred_subvert = layers.Reshape((1, self.hparams.filter_num))(pred_subvert)

        model = keras.Model(input_subvert, pred_subvert, name="subvert_encoder")
        return model

    def _build_naml(self):
        """The main function to create NAML's logic. The core of NAML
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
        """

        his_input_title = keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )
        his_input_body = keras.Input(
            shape=(self.hparams.history_size, self.hparams.body_size), dtype="int32"
        )
        his_input_vert = keras.Input(
            shape=(self.hparams.history_size, 1), dtype="int32"
        )
        his_input_subvert = keras.Input(
            shape=(self.hparams.history_size, 1), dtype="int32"
        )

        pred_input_title = keras.Input(
            # shape=(hparams.npratio + 1, hparams.title_size),
            shape=(None, self.hparams.title_size),
            dtype="int32",
        )
        pred_input_body = keras.Input(
            # shape=(hparams.npratio + 1, hparams.body_size),
            shape=(None, self.hparams.body_size),
            dtype="int32",
        )
        # pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        # pred_input_subvert = keras.Input(shape=(hparams.npratio + 1, 1), dtype="int32")
        pred_input_vert = keras.Input(shape=(None, 1), dtype="int32")
        pred_input_subvert = keras.Input(shape=(None, 1), dtype="int32")

        pred_input_title_one = keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="int32",
        )
        pred_input_body_one = keras.Input(
            shape=(
                1,
                self.hparams.body_size,
            ),
            dtype="int32",
        )
        pred_input_vert_one = keras.Input(shape=(1, 1), dtype="int32")
        pred_input_subvert_one = keras.Input(shape=(1, 1), dtype="int32")

        his_title_body_verts = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_body, his_input_vert, his_input_subvert]
        )

        pred_title_body_verts = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert]
        )

        pred_title_body_verts_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
            ]
        )
        pred_title_body_verts_one = layers.Reshape((-1,))(pred_title_body_verts_one)

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)

        user_present = self.userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        news_present_one = self.newsencoder(pred_title_body_verts_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model(
            [
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title,
                pred_input_body,
                pred_input_vert,
                pred_input_subvert,
            ],
            preds,
        )

        scorer = keras.Model(
            [
                his_input_title,
                his_input_body,
                his_input_vert,
                his_input_subvert,
                pred_input_title_one,
                pred_input_body_one,
                pred_input_vert_one,
                pred_input_subvert_one,
            ],
            pred_one,
        )

        return model, scorer
