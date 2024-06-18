from src.utils.get_training_args import get_training_args
from src.data.original_model_datamodule import OriginalModelDatamodule
from src.data.ebnerd_variants import EbnerdVariants

import wandb
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from src.model.original_lightning_module import OriginalModule
from src.model.components.model import Model
from transformers import BertTokenizer, BertModel

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


def main():
    args = get_training_args()
    
    wandb.login()

    data_download_path = EbnerdVariants.init_variant(args.ebnerd_variant).value.path

    datamodule = OriginalModelDatamodule(data_download_path=data_download_path, batch_size=args.batch_size, num_workers=args.num_workers, 
                                         api_key=args.api_key, history_size=args.history_size, fraction=args.fraction, npratio=args.npratio)

    datamodule.setup()
    train_news_title, train_news_entity, train_news_group = datamodule.data_train.get_word_ids(
        max_title_length=args.title_len, max_entity_length=40, max_group_length=40
    )
    val_news_title, val_news_entity, val_news_group = datamodule.data_val.get_word_ids(
        max_title_length=args.title_len, max_entity_length=40, max_group_length=40
    )
    # TODO: add test set

    

    #the last created dataset has the largest numeber
    n_users = datamodule.data_val.num_users #+ datamodule.test_set.n_users TODO: add test set
    train_user_news, train_news_user = datamodule.data_train.preprocess_neighbors()
    val_user_news, val_news_user = datamodule.data_val.preprocess_neighbors()
    net = Model(
        args,
        torch.tensor(train_news_title).to(device),
        torch.tensor(train_news_entity).to(device),
        torch.tensor(train_news_group).to(device),
        n_users
    )
    #datamodule.data_train.__getitem__()

    module = OriginalModule(net=net, args=args, train_user_news=train_user_news, train_news_user=train_news_user,
                            val_user_news=val_user_news, val_news_user=val_news_user, train_article_features=(torch.tensor(train_news_title), torch.tensor(train_news_entity), torch.tensor(train_news_group)),
                            val_article_features=(torch.tensor(val_news_title), torch.tensor(val_news_entity), torch.tensor(val_news_group)), n_users=n_users) #TODO add test set
    
    checkpoint_filename = f"{args.ebnerd_variant}-original-model"
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename + "-{epoch}-{val_loss:.2f}",
        monitor="val/loss",
        mode="min",
    )

    wandb_logger = WandbLogger(
        entity="inverse_rl", project="RecSys", name=checkpoint_filename
    )

    wandb_logger.watch(module, log="all")

    callbacks = [checkpoint_callback]

    trainer_args = {
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": wandb_logger,
        "accelerator": device_name,
        "devices": "auto",
        "limit_train_batches": 1
    }

    trainer = L.Trainer(**trainer_args)
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
