import requests
import zipfile
from pathlib import Path
import logging
from typing import Optional

import torch
from typing import Union
from torch import Tensor
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl
import numpy as np

from src.ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
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

    def __init__(self, root_dir, data_split, mode = "train", history_size = 30, fraction = 0.1):
        super().__init__()

        self.df_behaviors, self.df_history, articles = self.ebnerd_from_path(path=root_dir, history_size=history_size, mode=mode, data_split=data_split, fraction=fraction)

        self.unknown_representation = [0]

        #preprocess the articles into embedding vectors
        self.articles, self.article_mapping = self.preprocess_articles(articles)

        #something idk
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_mapping, unknown_representation=self.unknown_representation
        )

    def __len__(self):
        return len(self.df_behaviors)
    
    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        TODO: im really confused and idk if this is the best way to do things, but its something I can test atleast 
        """
        row = self.df_behaviors.row(idx)
       
        return row
    
    def preprocess_articles(self, df_articles: pl.DataFrame) -> pl.DataFrame:
        TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
        # this should be changed probably to be a parameter
        TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL] 
        MAX_TITLE_LENGTH = 30

        # LOAD HUGGINGFACE:
        transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
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



    def ebnerd_from_path(self, path: Path, mode: str, data_split, history_size: int = 30, fraction = 0.1) -> pl.DataFrame:
        """
        Load ebnerd - function
        # I could add something here to select columns but I dont think its necessary for now, makes more sense to do in the loader overwrite
        """
        print(f'processing {mode} data')
        if mode == "test":
            path = Path(path) / 'ebnerd_testset' / mode
            return None, None
        else:
            article_path = Path(path) / data_split / "articles.parquet"
            path = Path(path) / data_split / mode
        
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
        if mode == "test" or mode == "val":
            df_behaviors = (df_behaviors
                .pipe(create_binary_labels_column)
                .sample(fraction=fraction)
            )

        #also load article data 
        df_articles = pl.read_parquet(article_path)


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