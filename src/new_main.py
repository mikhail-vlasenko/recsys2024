import numpy as np

from src.ebrec.utils._behaviors import add_prediction_scores, add_known_user_column
from src.ebrec.utils._constants import DEFAULT_IMPRESSION_ID_COL,DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_USER_COL, DEFAULT_KNOWN_USER_COL
from src.utils.get_training_args import get_training_args
from src.utils.print_mean_std import print_mean_std
from src.data.original_model_datamodule import OriginalModelDatamodule
from src.data.ebnerd_variants import EbnerdVariants
from src.ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore, F1Score
from pathlib import Path

import wandb
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
import torch.functional as F
import polars as pl
from src.ebrec.utils._python import write_submission_file, rank_predictions_by_score
from lightning.pytorch.loggers import WandbLogger
from src.model.original_lightning_module import OriginalModule
from src.model.components.model import Model
from transformers import BertTokenizer, BertModel
from copy import copy 
import os
from torchmetrics import AUROC
from torchmetrics import F1Score as F1SCORE
from src.ebrec.evaluation import Coverage

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

def split_dataframe(df: pl.DataFrame, known_user_col: str = DEFAULT_KNOWN_USER_COL):
    """
    Splits the DataFrame into two: one with only known users and one with both known and unknown users.
    Args:
        df: A Polars DataFrame object with a column indicating known users.
        known_user_col: The name of the column indicating known users.
    Returns:
        A tuple of two DataFrames (known_users_df, all_users_df).
    """
    known_users_df = df.filter(pl.col(known_user_col) == True)
    all_users_df = df  # Since all_users_df is the same as the input df
    return known_users_df, all_users_df

def count_users(df: pl.DataFrame, known_user_col: str = DEFAULT_KNOWN_USER_COL):
    """
    Counts the number of known and unknown users in the DataFrame.
    Args:
        df: A Polars DataFrame object with a column indicating known users.
        known_user_col: The name of the column indicating known users.
    Returns:
        A tuple of two integers (num_known_users, num_unknown_users).
    """
    num_known_users = df.filter(pl.col(known_user_col) == True).height
    num_unknown_users = df.filter(pl.col(known_user_col) == False).height
    return num_known_users, num_unknown_users

def compute_catalog_coverage(df: pl.DataFrame, n_articles_test: int):
    """
    Computes the catalog coverage of the recommendations.
    Args:
        df: A Polars DataFrame object with a column of recommended articles.
        n_articles_test: The number of articles in the test set.
    Returns:
        The catalog coverage as a float.
    """
        # Function to sort articles by scores and get top 5
    def get_top_5_articles(articles, scores):
        sorted_indices = np.argsort(scores)[::-1]  # Indices of scores sorted in descending order
        top_5_indices = sorted_indices[:5]         # Get the top 5 indices
        top_5_articles = [articles[i] for i in top_5_indices]  # Get the articles corresponding to the top 5 indices
        return top_5_articles

    # Apply the function to each row
    recommended_articles_at_5_array = df.with_columns(
        pl.struct([DEFAULT_INVIEW_ARTICLES_COL, "scores"]).apply(lambda x: get_top_5_articles(x[DEFAULT_INVIEW_ARTICLES_COL], x["scores"])).alias("recommended_articles_at_5")
    )["recommended_articles_at_5"].to_list()

    #recommended_articles_at_5_array = df["inview_articles"].apply(lambda x: x.sort(descending=True)[:5]).to_list()
    print(recommended_articles_at_5_array)
    calculator = Coverage()

    coverage, coverage_frac = calculator(np.array(recommended_articles_at_5_array), np.arange(n_articles_test))
    print(f"Catalog coverage@5 frac: {coverage_frac:.2f}, catelog coverage: {coverage:.2f}")
    return coverage, coverage_frac
    

def train_and_test(data_download_path: str, args):
    datamodule = OriginalModelDatamodule(
        data_download_path=data_download_path, batch_size=args.batch_size, num_workers=args.num_workers,
        api_key=args.api_key, history_size=args.history_size, fraction=args.fraction, npratio=args.npratio,
        one_row_per_impression=args.one_row_impression, seed = args.seed, 
        use_labeled_test_set=args.use_labeled_test_set, 
        labeled_test_set_split = args.labeled_test_set_split,
        test_on_train=args.test_on_train
    )

    datamodule.setup()
    train_news_title, train_news_entity, train_news_group = datamodule.data_train.get_word_ids(
        max_title_length=args.title_len, max_entity_length=40, max_group_length=40
    )
    val_news_title, val_news_entity, val_news_group = datamodule.data_val.get_word_ids(
        max_title_length=args.title_len, max_entity_length=40, max_group_length=40
    )
    test_news_title, test_news_entity, test_news_group = datamodule.data_test.get_word_ids(
        max_title_length=args.title_len, max_entity_length=40, max_group_length=40
    )

    #the last created dataset has the largest numeber
    n_users = datamodule.data_test.num_users
    train_user_news, train_news_user = datamodule.data_train.preprocess_neighbors()
    val_user_news, val_news_user = datamodule.data_val.preprocess_neighbors()
    test_user_news, test_news_user = datamodule.data_test.preprocess_neighbors()
    net = Model(
        args,
        torch.tensor(train_news_title).to(device),
        torch.tensor(train_news_entity).to(device),
        torch.tensor(train_news_group).to(device),
        n_users
    )
    #datamodule.data_train.__getitem__()

    module = OriginalModule(net=net, args=args,
                            train_user_news=train_user_news, train_news_user=train_news_user,
                            val_user_news=val_user_news, val_news_user=val_news_user,
                            test_user_news=test_user_news, test_news_user=test_news_user,
                            train_article_features=(torch.tensor(train_news_title), torch.tensor(train_news_entity), torch.tensor(train_news_group)),
                            val_article_features=(torch.tensor(val_news_title), torch.tensor(val_news_entity), torch.tensor(val_news_group)),
                            test_article_features=(torch.tensor(test_news_title), torch.tensor(test_news_entity), torch.tensor(test_news_group)),
                            n_users=n_users,
                            val_df_behaviors=datamodule.data_val.behaviors_before_explode,
                            test_df_behaviors=datamodule.data_test.behaviors_before_explode)
    
    checkpoint_filename = f"{args.ebnerd_variant}-original-model"
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename + "-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min"
    )

    wandb_logger = WandbLogger(
        entity="inverse_rl", project="RecSys", config=vars(args)
    )

    wandb_logger.watch(module, log="all")

    callbacks = [checkpoint_callback]

    trainer_args = {
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": wandb_logger,
        "accelerator": device_name,
        "devices": "auto",
        'max_epochs': args.n_epochs
    }

    trainer = L.Trainer(**trainer_args)

    if args.checkpoint is not None:
        run = wandb.init(entity="inverse_rl", project="RecSys")
        current_checkpoint = run.use_artifact(args.checkpoint, type='model')
        directory = current_checkpoint.download()
        checkpoint = Path(directory) / os.listdir(directory)[0]
        module = OriginalModule.load_from_checkpoint(checkpoint, net=net)
    else:
        trainer.fit(module, datamodule)
        #load the best model
        checkpoint_path = checkpoint_callback.best_model_path
        artifact = wandb.Artifact('Weighting_model', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        #print(checkpoint_callback.best_model_path)
        #previous_module = copy.deepcopy(module)
        module = OriginalModule.load_from_checkpoint(checkpoint_callback.best_model_path, net=net)
        #test_sum = torch.sum(module.user_embedding - previous_module.user_embedding)

    trainer.test(module, datamodule)

    if args.flat_metrics:
        if args.test_on_train:
          labels = datamodule.data_train.df_behaviors["labels"].to_list()
        else:
          labels = datamodule.data_test.df_behaviors["labels"].to_list()
        print(labels)
        scores = module.test_predictions
        print(scores)
        auroc = AUROC(task="binary")
        aurmetric = auroc(torch.tensor(scores), torch.tensor(labels))
        print(aurmetric)
        f1 = F1SCORE(task="binary")
        f1metric = f1(torch.tensor(scores), torch.tensor(labels))
        print(f1metric)
        metrics = {'AUROC': aurmetric, 'F1': f1metric}
        print(metrics)
        wandb.finish()
        return metrics, None

    def revert_explosion(df, id_col, exploded_cols):
        # Identify columns that are not exploded
        
        all_exploded_cols = exploded_cols + [id_col]
    
        other_cols = [col for col in df.columns if col not in all_exploded_cols]

        # Group by the identifier column
        # For exploded columns, aggregate into lists
        # For other columns, take the first value (assuming they are identical within groups)
        #agg_exploded_cols = [pl.col(col) for col in exploded_cols]
        agg_other_cols = [pl.first(col).alias(col) for col in other_cols]
        df_reverted = df.groupby(id_col, maintain_order=True).agg(agg_other_cols ,article_ids_inview = pl.col('article_ids_inview'), labels = pl.col('labels'))# + agg_other_cols)

        return df_reverted

    if args.test_on_train:
      test_df: pl.DataFrame = revert_explosion(datamodule.data_train.df_behaviors, DEFAULT_IMPRESSION_ID_COL, ['article_ids_inview', 'labels']) #if type(datamodule.data_test.behaviors_before_explode) == pl.LazyFrame else datamodule.data_test.behaviors_before_explod
    else:
       test_df: pl.DataFrame = revert_explosion(datamodule.data_test.df_behaviors, DEFAULT_IMPRESSION_ID_COL, ['article_ids_inview', 'labels']) #if type(datamodule.data_test.behaviors_before_explode) == pl.LazyFrame else datamodule.data_test.behaviors_before_explod

    scores = np.array(module.test_predictions)[..., np.newaxis]
    
    
    test_df = add_prediction_scores(test_df, scores.tolist()).pipe(
        add_known_user_column, known_users=datamodule.data_train.df_behaviors[DEFAULT_USER_COL]
    )

    coverage, coverag_frac = compute_catalog_coverage(test_df, n_articles_test = datamodule.data_test.num_articles)

    metrics = None, None
    if args.use_labeled_test_set:
        test_df_known, test_df = split_dataframe(test_df)
        num_known_users, num_unknown_users = count_users(test_df)
        print('count of known users', num_known_users)
        print('count of unknown users', num_unknown_users)

        known_labels = test_df_known["labels"].to_list()
        known_scores = test_df_known["scores"].to_list()
        labels = test_df["labels"].to_list()
        scores = test_df["scores"].to_list()

        print(test_df_known.head(10))
        metrics_known = MetricEvaluator(
            labels=known_labels,
            predictions=known_scores,
            metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10), F1Score(threshold=0.5)],
        )

        # trainer.fit(module, datamodule)
        metrics = MetricEvaluator(
            labels=labels,
            predictions=scores,
            metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10), F1Score(threshold=0.5)],
        )

        metrics = metrics.evaluate().evaluations
        metrics_known = metrics_known.evaluate().evaluations
        print('metrics for all users', metrics)
        print('metrics for only known users', metrics_known)

        metrics = metrics, metrics_known

    else:
        test_df = test_df.with_columns(
            pl.col("scores")
            .map_elements(lambda x: list(rank_predictions_by_score(x)))
            .alias("ranked_scores")
        )
        write_submission_file(
            impression_ids=test_df[DEFAULT_IMPRESSION_ID_COL],
            prediction_scores=test_df["ranked_scores"],
        )

    wandb.finish()

    metrics_coverage = {'coverage': coverage, 'coverage_frac': coverag_frac}
    metrics0 = metrics[0]
    metrics1 = metrics[1]

    metrics0 = {**metrics[0], **metrics_coverage}
    metrics1 = {**metrics[1], **metrics_coverage}
    print(metrics0)
    print(metrics_coverage)
    return metrics0, metrics1

def main():
    args = get_training_args()
    
    try:
        wandb.login()
    except:
        with open("src/utils/wandb_api_key.txt") as f:
            wandb_api_key = f.read().strip()
            wandb.login(key=wandb_api_key)

    data_download_path = EbnerdVariants.init_variant(args.ebnerd_variant).value.path
    metrics_list = []
    known_metrics_list = []
    for i in range(args.num_runs):
        #set seed 
        seed = args.seeds[i]
        args.seed = seed
        L.seed_everything(seed)
        if args.checkpoint_list is not None:
            args.checkpoint = args.checkpoint_list[i]
        else:
            args.checkpoint = None

        metrics, known_metrics = train_and_test(data_download_path=data_download_path, args=args)
        metrics_list.append(metrics)
        known_metrics_list.append(known_metrics)

    if args.use_labeled_test_set:
        print_mean_std(metrics_list)
        print_mean_std(known_metrics_list)

if __name__ == "__main__":
    main()
