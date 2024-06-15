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


def main():
    args = get_training_args()
    
    wandb.login()

    data_download_path = EbnerdVariants.init_variant(args.ebnerd_variant).value.path

    datamodule = OriginalModelDatamodule(data_download_path=data_download_path, batch_size=args.batch_size, num_workers=args.num_workers, api_key=args.api_key)

    datamodule.setup()
    news_title, news_entity, news_group, id_to_index = datamodule.data_train.get_word_ids(max_title_length=args.title_len)
    n_users = datamodule.data_train.get_n_users()
    n_news = len(news_title)
    train_user_news, train_news_user = datamodule.data_train.preprocess_neighbors()
    #datamodule.data_train.__getitem__()
    
    net = Model(args, torch.tensor(news_title), torch.tensor(news_entity), torch.tensor(news_group), n_users, len(news_title))

    module = OriginalModule(net, compile=True, args=args, train_user_news=train_user_news, train_news_user=train_news_user, n_news=n_news)
    checkpoint_filename = f"{args.ebnerd_variant}-original-model"
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename + "-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
    )

    wandb_logger = WandbLogger(
        entity="inverse-rl", project="RecSys", name=checkpoint_filename
    )

    wandb_logger.watch(module, log="all")

    callbacks = [checkpoint_callback]

    trainer_args = {
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "logger": wandb_logger,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto"
    }

    trainer = L.Trainer(**trainer_args)
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()