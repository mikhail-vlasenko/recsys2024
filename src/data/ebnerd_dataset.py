import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional

from polars import DataFrame
from tqdm import tqdm

import torch
from typing import Union
from torch import Tensor
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import polars as pl
import numpy as np
import os
import pickle 

from src.ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_NER_COL,
    DEFAULT_ENTITIES_COL, DEFAULT_ARTICLE_ID_COL
)

from src.ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from src.ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from src.ebrec.utils._articles import convert_text2encoding_with_transformers
from src.ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from src.ebrec.utils._articles import create_article_id_to_value_mapping
from src.ebrec.utils._nlp import get_transformers_word_embeddings
from src.ebrec.utils._python import write_submission_file, rank_predictions_by_score

from src.ebrec.models.newsrec.dataloader import NRMSDataLoader
from src.ebrec.models.newsrec.model_config import hparams_nrms
from src.ebrec.models.newsrec import NRMSModel

from src.ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

class EbnerdDataset(Dataset):

    def __init__(self, root_dir, data_split, mode = "train", history_size = 30, fraction = 1, seed = 0, npratio=4, user_id_to_index=None, article_id_to_index=None):
        """
        User_id_to_index and article_id_to_index are in this constructor because they can be passed to consectutive datasets
        This is useful when we want to use the same mapping for the train, validation and test sets. 
        That means that for validation we start from the given input, and add the new users and articles to the mapping.

        self.num_users and self.num_articles will still refer to the num_users for the current dataset
        """
        super().__init__()

        self.df_behaviors: DataFrame
        self.df_behaviors, self.df_history, self.article_df = self.ebnerd_from_path(path=root_dir, history_size=history_size, mode=mode, data_split=data_split, fraction=fraction, seed=seed, npratio=npratio)

        self.num_users: int
        self.num_articles: int
        self.user_id_to_index = self.compress_user_ids(user_id_to_index=user_id_to_index)
        self.article_id_to_index = self.compress_article_ids(article_id_to_index=article_id_to_index)

        #assert max(self.df_behaviors[DEFAULT_USER_COL]) + 1 == len(self.df_behaviors[DEFAULT_USER_COL].unique()), "User ids are not continuous"

    def __len__(self):
        return len(self.df_behaviors)
    
    def __getitem__(self, idx) -> tuple[int, int, int]:
        row = self.df_behaviors.row(named=True, index=idx)

        # Get the required columns 
        user_id = row['user_id'] #DEFAULT_USER_COL = "user_id"
        article_ids_clicked = row['article_ids_inview'] #DEFAULT_INVIEW_ARTICLES_COL
        labels = row['labels'] #DEFAULT_LABELS_COL

        return user_id, article_ids_clicked, labels
    
    def compress_user_ids(self, user_id_to_index=None) -> dict[int, int]:

        if user_id_to_index is None:
            # Get the unique user ids
            unique_user_ids = self.df_behaviors[DEFAULT_USER_COL].unique().to_numpy()
            
            # Create a mapping from user id to index
            user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
            self.num_users = len(user_id_to_index)
        else:
            current_unique_user_ids = self.df_behaviors[DEFAULT_USER_COL].unique().to_numpy()
            previous_unique_user_ids = np.array(list(user_id_to_index.keys()))
            unique_user_ids = np.unique(np.concatenate([current_unique_user_ids, previous_unique_user_ids]))
            
            # Create a mapping from user id to index
            user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
            self.num_users = len(user_id_to_index)

        # Replace the user ids with the index
        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_USER_COL).apply(lambda user_id: user_id_to_index[user_id]).alias(DEFAULT_USER_COL)
        )

        return user_id_to_index

    def compress_article_ids(self, article_id_to_index=None) -> dict[int, int]:
        
        if article_id_to_index is None:
            unique_article_ids = self.article_df[DEFAULT_ARTICLE_ID_COL].unique().to_numpy()
            article_id_to_index = {user_id: index for index, user_id in enumerate(unique_article_ids)}
            self.num_articles = len(article_id_to_index)
        else:
            current_unique_article_ids = self.article_df[DEFAULT_ARTICLE_ID_COL].unique().to_numpy()
            previous_unique_article_ids = np.array(list(article_id_to_index.keys()))
            unique_article_ids = np.unique(np.concatenate([current_unique_article_ids, previous_unique_article_ids]))

            article_id_to_index = {article_id: index for index, article_id in enumerate(unique_article_ids)}
            self.num_articles = len(article_id_to_index)

        article_id_to_index[np.nan] = np.nan

        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_ARTICLE_ID_COL).apply(lambda article_id: article_id_to_index[int(article_id)]).alias(DEFAULT_ARTICLE_ID_COL)
        )
        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_INVIEW_ARTICLES_COL).apply(
                lambda article_id: article_id_to_index[int(article_id)]
            ).alias(DEFAULT_INVIEW_ARTICLES_COL)
        )
        self.article_df = self.article_df.with_columns(
            pl.col(DEFAULT_ARTICLE_ID_COL).apply(lambda article_id: article_id_to_index[int(article_id)]).alias(DEFAULT_ARTICLE_ID_COL)
        )

        return article_id_to_index
    
    def get_word_ids(self, max_title_length, max_entity_length, max_group_length):
        '''
        Return ids for the words in the title, the entity groups and the named entities in the text.
        To follow the original paper, named entities are encoded as words using the same ids as the words in the title,
        while the entity groups are tokenized separately.
        
        For the title use 
        DEFAULT_TITLE_COL = "title" -> "Hanks beskyldt…"
        For the entity groups use
        DEFAULT_ENTITIES_COL = "entity_groups" -> ["PER"]
        for the named entities use
        DEFAULT_NER_COL = "ner_clusters" -> ["David Gardner"]
        '''
        print("getting word ids")
        #intialize the tokenizer
        TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"

        transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

        #encode the titles 
        titles_list= self.article_df[DEFAULT_TITLE_COL].to_list()
        encoding = transformer_tokenizer(titles_list, return_tensors='pt', padding='max_length', max_length=max_title_length, truncation=True)
        title_word_ids = encoding['input_ids']

        #encode the named entities
        entities_list = self.article_df[DEFAULT_ENTITIES_COL].to_list()
        placeholder = ['[UNK]']  
        prepared_entities = [ent if ent else placeholder for ent in entities_list]
        # print(prepared_entities)
        encoding = transformer_tokenizer(prepared_entities, return_tensors='pt', padding='longest', truncation=True, is_split_into_words =True, max_length=max_entity_length)
        entities_word_ids = encoding['input_ids']

        #encode the entity groups
        entity_group_list = self.article_df[DEFAULT_ENTITIES_COL].to_list()
        entity_group_dict = self.build_dictionary(entity_group_list)
        ner_word_ids = self.texts_to_id(entity_group_list, entity_group_dict, max_group_length)

        return title_word_ids, entities_word_ids, ner_word_ids
    
    def build_dictionary(self, texts: list[list[str]]):
        """
        Build a dictionary from a list of lists of words.
        The dictionary maps each word to a unique integer.
        """
        unique_words = set()
        for text in tqdm(texts):
            unique_words.update(text)
        word_dict = {'[PAD]': 0} 
        for word in tqdm(unique_words):
            if word not in word_dict:
                word_dict[word] = len(word_dict)
        return word_dict

    def texts_to_id(self, texts: list[list[str]], word_dict: dict, max_length: int):
        return [
            (tokens := [word_dict.get(word, word_dict['[PAD]']) for word in text])[:max_length] +
            [word_dict['[PAD]']] * (max_length - len(tokens))
            if text else [word_dict['[PAD]']] * max_length  #handle empty lists with full padding
            for text in texts
        ]
    
    def preprocess_neighbors(self):
        news_user = [[] for _ in range(self.num_articles)]
        user_news = [[] for _ in range(self.num_users)]
        for row in self.df_behaviors.rows(named=True):
            news_id = row[DEFAULT_INVIEW_ARTICLES_COL]
            user_id = row[DEFAULT_USER_COL]
        
            if user_id not in news_user[news_id]:
                news_user[news_id].append(user_id)
            if news_id not in user_news[user_id]:
                user_news[user_id].append(news_id)

        for list1 in news_user:
            if not list1:
                list1.append(0)
        for list2 in user_news:
            if not list2:
                list2.append(0)

        return user_news, news_user

    def add_clicked_articles_column(self,
        df: pl.DataFrame,
        clicked_col: str = DEFAULT_CLICKED_ARTICLES_COL
    ) -> pl.DataFrame:
        df_height = df.height
        df = df.lazy()
        none_list_series = pl.Series(clicked_col, [None] * df_height, dtype=pl.List(pl.Int64))
        return df.with_columns(none_list_series), df_height

    def ebnerd_from_path(self, path: Path, mode: str, data_split, seed, npratio, history_size: int = 30, fraction = 1) -> tuple[pl.DataFrame, pl.LazyFrame, pl.DataFrame]:
        """
        Load ebnerd - function
        # I could add something here to select columns but I dont think its necessary for now, makes more sense to do in the loader overwrite
        """
        print(f'processing {mode} data')
        if mode == "test":
            article_path = Path(path) / "ebnerd_testset/articles.parquet"
            path = Path(path) / 'ebnerd_testset' / mode

        else:
            article_path = Path(path) / data_split / "articles.parquet"
            path = Path(path) / data_split / mode

        data_pkl_path = Path('data') / f'{mode}_seed_{seed}.pkl'

        if os.path.exists(data_pkl_path):
            with open(data_pkl_path, 'rb') as f:
                (df_behaviors, df_history, df_articles) = pickle.load(f)

            return df_behaviors, df_history, df_articles

        else:
            df_history = (
                pl.scan_parquet(path.joinpath("history.parquet"))
                .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
                .pipe(
                    truncate_history,
                    column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                    history_size=history_size,
                    padding_value=0,
                    enable_warning=False,
                )
            )
            df_behaviors = (
                pl.scan_parquet(path.joinpath("behaviors.parquet"))
                .collect()
                .pipe(
                    slice_join_dataframes,
                    df2=df_history.collect(),
                    on=DEFAULT_USER_COL,
                    how="left",
                )
            )

            if mode == "train":
                df_behaviors = (df_behaviors
                    .pipe(
                        sampling_strategy_wu2019,
                        npratio=npratio,
                        shuffle=True,
                        with_replacement=True,
                        seed=123,
                    )
                    .pipe(create_binary_labels_column)
                    .sample(fraction=fraction)
                )
            if mode == "validation":
                df_behaviors = (df_behaviors
                    .pipe(create_binary_labels_column)
                    .sample(fraction=fraction)
                )
            if mode == "test":
                df_behaviors, df_height = self.add_clicked_articles_column(df_behaviors)
                sample_size = fraction * df_height
                df_behaviors = (df_behaviors
                    .pipe(create_binary_labels_column)
                    .fetch(n_rows=sample_size)
                )

            #unroll the inview column as rows into the dataframe
            print(f'Exploding inview and labels columns...')
            df_behaviors = df_behaviors.explode('article_ids_inview','labels')

            #also load article data
            print(f'Loading article data...')
            df_articles = pl.read_parquet(article_path)

            #pickle the data
            print(f'Pickling data to {data_pkl_path}...')
            with open(data_pkl_path, 'wb') as f:
                pickle.dump((df_behaviors,
                             df_history.collect(),
                             df_articles),
                             f)

            print(f'Processing completed successfully')
            return df_behaviors, df_history, df_articles

    @classmethod
    def download_and_extract(cls, root_dir: str, data_download_path: str, api_key: str):
        zipfile_name = data_download_path.split("/")[-1].split("?")[0]
        folder_name = data_download_path.split("/")[-1].split(".")[0]

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        root_dir = Path(root_dir)
        data_dir = root_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        out_folder = data_dir / folder_name
        if out_folder.exists():
            print(f"Data already downloaded and extracted at {out_folder}")
            return out_folder

        print(f"Downloading and extracting data to {out_folder}")

        r = requests.get(data_download_path, headers=headers, allow_redirects=True)

        with open(data_dir / zipfile_name, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(data_dir / zipfile_name, "r") as zip_ref:
            zip_ref.extractall(out_folder)
        # remove the zip file
        (data_dir / zipfile_name).unlink()

        #also download the test set
        test_set_path = "https://huggingface.co/datasets/recsys2024/ebnerd/resolve/main/ebnerd_testset.zip?download=true"
        r = requests.get(test_set_path, headers=headers, allow_redirects=True)
        with open(data_dir / "ebnerd_testset.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(data_dir / "ebnerd_testset.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)
        # remove the zip file
        (data_dir / "ebnerd_testset.zip").unlink()

        return out_folder