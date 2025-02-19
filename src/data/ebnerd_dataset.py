import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional, Union, Any
from collections import defaultdict

from polars import DataFrame
from tqdm import tqdm

import torch
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
    DEFAULT_ENTITIES_COL, DEFAULT_ARTICLE_ID_COL, DEFAULT_SCROLL_PERCENTAGE_COL, DEFAULT_READ_TIME_COL,
    DEFAULT_KNOWN_USER_COL
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
    def __init__(
            self, root_dir, data_split,
            mode="train", history_size=30, fraction=1, seed=0, npratio=4, one_row_per_impression=False,
            user_id_to_index=None, article_id_to_index=None, train_df_behaviors=None, data_slice=None):
        """
        User_id_to_index and article_id_to_index are in this constructor because they can be passed to consectutive datasets
        This is useful when we want to use the same mapping for the train, validation and test sets.
        That means that for validation we start from the given input, and add the new users and articles to the mapping.

        self.num_users and self.num_articles will still refer to the num_users for the current dataset
        """
        super().__init__()

        self.mode = mode
        #self.first_n_test_rows = 1000
        self.max_inview_articles_at_test_time = 100

        if train_df_behaviors is None:
            self.df_behaviors: DataFrame
            self.df_behaviors, _, self.article_df, _ = self.ebnerd_from_path(path=root_dir, history_size=history_size, mode=self.mode, data_split=data_split, fraction=fraction, seed=seed, npratio=npratio, one_row_per_impression=one_row_per_impression)
        else:
            # if mode is test or val we still need to use the train_df_behaviors for the edges
            self.df_behaviors, _, self.article_df, self.behaviors_before_explode = self.ebnerd_from_path(path=root_dir, history_size=history_size, mode=self.mode, data_split=data_split, fraction=fraction, seed=seed, npratio=npratio, one_row_per_impression=False, data_slice=data_slice)

        self.num_users: int
        self.num_articles: int

        self.user_id_to_index = self.compress_user_ids(user_id_to_index=user_id_to_index)
        self.article_id_to_index = self.compress_article_ids(article_id_to_index=article_id_to_index)

        if self.mode == "train":
            self.train_df_behaviors = self.df_behaviors
        else:
            self.train_df_behaviors = train_df_behaviors

        #assert max(self.df_behaviors[DEFAULT_USER_COL]) + 1 == len(self.df_behaviors[DEFAULT_USER_COL].unique()), "User ids are not continuous"

    def __len__(self):
        return len(self.df_behaviors)
    
    def __getitem__(self, idx) -> tuple[int, Any, int]:
        row = self.df_behaviors.row(named=True, index=idx)

        # Get the required columns 
        user_id = row[DEFAULT_USER_COL]
        article_ids_clicked = row[DEFAULT_INVIEW_ARTICLES_COL]
        if self.mode == "test":
            labels = 0 #making it none is not allowed
        else:
            labels = row[DEFAULT_LABELS_COL]

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
            for user_id in current_unique_user_ids:
                if user_id not in user_id_to_index:
                    user_id_to_index[user_id] = len(user_id_to_index)
            self.num_users = len(user_id_to_index)

        # Replace the user ids with the index
        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_USER_COL).apply(lambda user_id: user_id_to_index[user_id]).alias(DEFAULT_USER_COL)
        )

        return user_id_to_index

    def compress_article_ids(self, article_id_to_index=None, news_user=None) -> dict[int, int]:
        if article_id_to_index is None:
            unique_article_ids = self.article_df[DEFAULT_ARTICLE_ID_COL].unique().to_numpy()
            article_id_to_index = {user_id: index for index, user_id in enumerate(unique_article_ids)}
            self.num_articles = len(article_id_to_index)
        else:
            current_unique_article_ids = self.article_df[DEFAULT_ARTICLE_ID_COL].unique().to_numpy()
            for article_id in current_unique_article_ids:
                if article_id not in article_id_to_index:
                    article_id_to_index[article_id] = len(article_id_to_index)
            self.num_articles = len(article_id_to_index)

        article_id_to_index[np.nan] = np.nan

        def replace_column(name, replace_list):
            if replace_list:
                func = lambda article_ids: [article_id_to_index[int(article_id)] for article_id in article_ids]
            else:
                func = lambda article_id: article_id_to_index[int(article_id)]
            self.df_behaviors = self.df_behaviors.with_columns(
                pl.col(name).apply(func).alias(name)
            )
        if self.mode == "train" or self.mode == "validation":
            replace_column(DEFAULT_ARTICLE_ID_COL, False )
            replace_column(DEFAULT_CLICKED_ARTICLES_COL, True)
            #we should only use article clicked if in train mode
            replace_column(DEFAULT_INVIEW_ARTICLES_COL, False)
        if self.mode == "test":
            replace_column(DEFAULT_INVIEW_ARTICLES_COL, False)

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
        # print(title_word_ids)

        #encode the named entities
        entities_list = self.article_df[DEFAULT_NER_COL].to_list()
        placeholder = ['[UNK]']  
        prepared_entities = [ent if ent else placeholder for ent in entities_list]
        # print(prepared_entities)
        encoding = transformer_tokenizer(prepared_entities, return_tensors='pt', padding='longest', truncation=True, is_split_into_words =True, max_length=max_entity_length)
        entities_word_ids = encoding['input_ids']
        # print(self.mode)
        # print(entities_word_ids)

        #encode the entity groups
        entity_group_list = self.article_df[DEFAULT_ENTITIES_COL].to_list()
        #entity_group_dict = self.build_dictionary(entity_group_list)
        entity_group_dict = {'[PAD]': 0, 'EVENT': 1, 'ORG': 2, 'PER': 3, 'MISC': 4, 'PROD': 5, 'LOC': 6}
        ner_word_ids = self.texts_to_id(entity_group_list, entity_group_dict, max_group_length)
        ner_word_ids = torch.tensor(ner_word_ids)
        #print(entity_group_list)
        #print(entity_group_dict)
        print(ner_word_ids)

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

        for row in self.train_df_behaviors.rows(named=True):
            news_ids = row[DEFAULT_CLICKED_ARTICLES_COL]
            user_id = row[DEFAULT_USER_COL]
            # so teeeechnically if we have multiple articles in news_ids,
            # we should give them different scroll percentages and read times
            # but the train set has at most 2 of those scroll percentages and read times per list,
            # and the test set has just one, so we'll just use the first one for all articles
            # (most of them are singular anyway)
            read_time = row[DEFAULT_READ_TIME_COL]
            scroll_percentage = row[DEFAULT_SCROLL_PERCENTAGE_COL]


            for news_id in news_ids:
                #print(news_id)
                if user_id not in news_user[news_id]:
                    news_user[news_id].append([user_id, read_time, scroll_percentage])
                if news_id not in user_news[user_id]:
                        user_news[user_id].append([news_id, read_time, scroll_percentage])

        for list1 in news_user:
            if not list1:
                list1.append([0, 0, 0])
        for list2 in user_news:
            if not list2:
                list2.append([0, 0, 0])

        return user_news, news_user

    def ebnerd_from_path(
            self, path: Path, mode: str, data_split, seed, npratio,
            history_size: int = 30, fraction=1, one_row_per_impression=False, data_slice=None
    ):
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

        data_pkl_path = Path('data') / f'{mode}_seed_{seed}'

        if data_slice is not None:
            start, end = data_slice
            print(f"Using data slice {start} to {end}")
            start = (100*start)
            end = (100*end)
            data_pkl_path = data_pkl_path.with_name(f"{data_pkl_path.name}_slice_{start}_{end}")

        data_pkl_path = data_pkl_path.with_suffix('.pkl')

        if os.path.exists(data_pkl_path):
            print(f"\nLoading data from {data_pkl_path}\n")
            with open(data_pkl_path, 'rb') as f:
                (df_behaviors, df_history, df_articles, df_before_explode) = pickle.load(f)

            return df_behaviors, df_history, df_articles, df_before_explode

        else:
            if self.mode == "test":
                df_history = (
                    pl.scan_parquet(path.joinpath("history.parquet"))
                    .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
                )
                df_behaviors = (
                    pl.scan_parquet(path.joinpath("behaviors.parquet"))
                    .collect()
                    .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_READ_TIME_COL, DEFAULT_SCROLL_PERCENTAGE_COL,DEFAULT_IMPRESSION_ID_COL)
                    .pipe(
                        slice_join_dataframes,
                        df2=df_history.collect(),
                        on=DEFAULT_USER_COL,
                        how="left",
                    )
                )
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
                    .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_READ_TIME_COL, DEFAULT_SCROLL_PERCENTAGE_COL, DEFAULT_ARTICLE_ID_COL, DEFAULT_CLICKED_ARTICLES_COL,DEFAULT_IMPRESSION_ID_COL)
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
                    seed=seed,
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
                df_behaviors = df_behaviors.head(1000000)
                #df_behaviors = df_behaviors.sample(fraction=fraction)

            if data_slice is not None:
                start, end = data_slice
                start_index = int(start * len(df_behaviors))
                length = int((end - start) * len(df_behaviors))
                df_behaviors = df_behaviors.slice(start_index, length) 

            behaviors_before_explode = df_behaviors

            print(f"Loaded {len(df_behaviors)} rows")
            if one_row_per_impression:
                # keep only one row per impression id
                df_behaviors = df_behaviors.unique(subset=DEFAULT_IMPRESSION_ID_COL, keep="first")
                print(f"Kept one row per impression id, now {len(df_behaviors)} rows")

            print(f'Exploding inview and labels columns...')
            if mode == "test":
                df_behaviors = df_behaviors.explode('article_ids_inview')
            else:
                df_behaviors = df_behaviors.explode('article_ids_inview','labels')

            #print the percentage of positive versus negative labels in the val and train datasets
            if mode == "train" or mode == "validation":
                print(f"Percentage of positive labels in {mode} data: {df_behaviors.filter(pl.col('labels') == 1).height / df_behaviors.height}")

            #also load article data
            df_articles = pl.read_parquet(article_path)

            #pickle the data
            print(f'Pickling data to {data_pkl_path}...')
            with open(data_pkl_path, 'wb') as f:
                pickle.dump((df_behaviors, df_history.collect(), df_articles, behaviors_before_explode), f)

            print(f'Processing completed successfully')
            return df_behaviors, df_history, df_articles, behaviors_before_explode

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