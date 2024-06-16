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

    def __init__(self, root_dir, data_split, mode = "train", history_size = 30, fraction = 1, seed = 0):
        super().__init__()

        self.df_behaviors: DataFrame
        self.df_behaviors, self.df_history, self.article_df = self.ebnerd_from_path(path=root_dir, history_size=history_size, mode=mode, data_split=data_split, fraction=fraction, seed=seed)

        self.num_users: int
        self.num_articles: int
        self.compress_user_ids()
        self.compress_article_ids()
        assert max(self.df_behaviors[DEFAULT_USER_COL]) + 1 == len(self.df_behaviors[DEFAULT_USER_COL].unique()), "User ids are not continuous"

        self.unknown_representation = "zeros"

        #preprocess the articles into embedding vectors
        #self.embedded_articles, self.article_mapping = self.preprocess_articles(articles)

        #something idk
        #self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
        #    self.article_mapping, unknown_representation=self.unknown_representation
        #)

    def __len__(self):
        return len(self.df_behaviors)
    
    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        row = self.df_behaviors.slice(idx, 1)

        # Get the required columns and convert them to numpy arrays if needed
        user_id = row['user_id'][0]
        # i think we should use DEFAULT_ARTICLE_ID_COL instead
        article_ids_clicked = row['article_ids_clicked'][-1][0] #idk if this can return more than one element. If so idk how to deal with it. 
        labels = row['labels'][0][-1]

        # Return the tuple
        #print(article_ids_clicked)
        return user_id, article_ids_clicked, labels
    
    def compress_user_ids(self):
        # Get the unique user ids
        unique_user_ids = self.df_behaviors[DEFAULT_USER_COL].unique().to_numpy()
        # Create a mapping from user id to index
        user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
        self.num_users = len(user_id_to_index)
        # Replace the user ids with the index
        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_USER_COL).apply(lambda user_id: user_id_to_index[user_id]).alias(DEFAULT_USER_COL)
        )

    def compress_article_ids(self):
        unique_article_ids = self.article_df[DEFAULT_ARTICLE_ID_COL].unique().to_numpy()

        article_id_to_index = {user_id: index for index, user_id in enumerate(unique_article_ids)}
        self.num_articles = len(article_id_to_index)
        article_id_to_index[np.nan] = np.nan

        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_ARTICLE_ID_COL).apply(lambda article_id: article_id_to_index[int(article_id)]).alias(DEFAULT_ARTICLE_ID_COL)
        )
        self.df_behaviors = self.df_behaviors.with_columns(
            pl.col(DEFAULT_CLICKED_ARTICLES_COL).apply(
                lambda article_ids: [article_id_to_index[int(article_id)] for article_id in article_ids]
            ).alias(DEFAULT_CLICKED_ARTICLES_COL)
        )
        self.article_df = self.article_df.with_columns(
            pl.col(DEFAULT_ARTICLE_ID_COL).apply(lambda article_id: article_id_to_index[int(article_id)]).alias(DEFAULT_ARTICLE_ID_COL)
        )

    def get_n_users(self) -> int:
        return len(self.df_behaviors[DEFAULT_USER_COL])
    
    def get_word_ids(self, max_title_length, max_entity_length, max_group_length) -> Tensor:
        print("getting word ids")
        #intialize the tokenizer
        TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
        TEXT_COLUMNS_TO_USE = [DEFAULT_TITLE_COL, DEFAULT_ENTITIES_COL, DEFAULT_NER_COL]

        transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

        #encode the titles 
        titles_list= self.article_df[DEFAULT_TITLE_COL].to_list()
        # print(titles_list)
        encoding = transformer_tokenizer(titles_list, return_tensors='pt', padding='max_length', max_length=max_title_length, truncation=True)
        title_word_ids = encoding['input_ids']

        #encode the enities
        entities_list = self.article_df[DEFAULT_ENTITIES_COL].to_list()
        placeholder = ['[UNK]']  
        prepared_entities = [ent if ent else placeholder for ent in entities_list]
        # print(prepared_entities)
        encoding = transformer_tokenizer(prepared_entities, return_tensors='pt', padding='longest', truncation=True, is_split_into_words =True, max_length=max_entity_length)
        entities_word_ids = encoding['input_ids']

        #encode the ner
        # todo: whats ner?
        ner_list = self.article_df[DEFAULT_NER_COL].to_list()
        ner_dict = self.build_dictionary(ner_list)
        ner_word_ids = self.tokenize_texts(ner_list, ner_dict, max_group_length)

        return title_word_ids, entities_word_ids, ner_word_ids
    
    def build_dictionary(self, texts):
        unique_words = set()
        for text in tqdm(texts):
            unique_words.update(text)  # Assuming text is a list of words
        word_dict = {'[PAD]': 0}  # Initialize with padding token
        for word in tqdm(unique_words):
            if word not in word_dict:  # This check prevents overwriting existing tokens
                word_dict[word] = len(word_dict)
        return word_dict

    def tokenize_texts(self, texts, word_dict, max_length):
        return [
            (tokens := [word_dict.get(word, word_dict['[PAD]']) for word in text])[:max_length] +
            [word_dict['[PAD]']] * (max_length - len(tokens))
            if text else [word_dict['[PAD]']] * max_length  # Handle empty lists with full padding
            for text in texts
        ]
    
    def preprocess_neighbors(self):
        news_user = [[] for _ in range(self.num_articles)]
        user_news = [[] for _ in range(self.num_users)]
        for row in self.df_behaviors.rows(named=True):
            news_ids = row[DEFAULT_CLICKED_ARTICLES_COL]
            user_id = row[DEFAULT_USER_COL]
            for news_id in news_ids:
                if user_id not in news_user[news_id]:
                    news_user[news_id].append(user_id)
                if news_id not in user_news[user_id]:
                    user_news[user_id].append(news_id)

        return user_news, news_user
    
    def preprocess_articles(self, df_articles: pl.DataFrame) -> pl.DataFrame:
        TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
        # this should be changed probably to be a parameter
        TEXT_COLUMNS_TO_USE = [DEFAULT_TITLE_COL, DEFAULT_ENTITIES_COL, DEFAULT_NER_COL] 
        MAX_TITLE_LENGTH = 30

        # LOAD HUGGINGFACE:
        #transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
        transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

        df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
        df_articles, token_col_title = convert_text2encoding_with_transformers(
            df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
        )

        # =>
        article_mapping = create_article_id_to_value_mapping(
            df=df_articles, value_col=token_col_title
        )

        return df_articles, article_mapping

    def ebnerd_from_path(self, path: Path, mode: str, data_split, seed, history_size: int = 30, fraction = 1) -> pl.DataFrame:
        """
        Load ebnerd - function
        # I could add something here to select columns but I dont think its necessary for now, makes more sense to do in the loader overwrite
        """
        print(f'processing {mode} data')
        if mode == "test":
            path = Path(path) / 'ebnerd_testset' / mode
            return None, None, None
        else:
            article_path = Path(path) / data_split / "articles.parquet"
            path = Path(path) / data_split / mode

        data_pkl_path = Path('data') / f'{mode}_seed_{seed}.pkl'

        if os.path.exists(data_pkl_path):
            with open(data_pkl_path, 'rb') as f:
                (df_behaviors, df_history, df_articles) = pickle.load(f)

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
                        npratio=4,
                        shuffle=True,
                        with_replacement=True,
                        seed=123,
                    )
                    .pipe(create_binary_labels_column)
                    .sample(fraction=fraction)
                )
            if mode == "test" or mode == "validation":
                df_behaviors = (df_behaviors
                    .pipe(create_binary_labels_column)
                    .sample(fraction=fraction)
                )

            #also load article data 
            df_articles = pl.read_parquet(article_path)

            #pickle the data
            with open(data_pkl_path, 'wb') as f:
                pickle.dump((df_behaviors.collect(), df_history.collect(), df_articles.collect()), f)


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