from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.ebnerd_dataset import EbnerdDataset
from src.ebrec.models.newsrec.dataloader import NewsrecDataLoader


class OriginalModelDatamodule(LightningDataModule):
    def __init__(
        self,
        data_download_path: str,
        api_key: str,
        root_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
        history_size: int = 30,
        fraction: float = 1.0,
        npratio: int = 4,
        one_row_per_impression: bool = False,
    ) -> None:
        super().__init__()
    
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_split = data_download_path.split("/")[-1].split(".")[0]

        # data transformations
        self.data_train: Optional[EbnerdDataset] = None
        self.data_val: Optional[EbnerdDataset] = None
        self.data_test: Optional[EbnerdDataset] = None

        self.batch_size_per_device = batch_size
    
    def prepare_data(self) -> None:
        EbnerdDataset.download_and_extract(root_dir=self.hparams.root_dir, data_download_path=self.hparams.data_download_path, api_key=self.hparams.api_key)

    def setup(self, stage: Optional[str] = None) -> None:
        self.prepare_data()
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_params = {
                "root_dir": self.hparams.root_dir,
                "data_split": self.data_split,
                "history_size": self.hparams.history_size,
                "fraction": self.hparams.fraction,
                "npratio": self.hparams.npratio,
            }
            self.data_train: Optional[EbnerdDataset] = EbnerdDataset(
                mode="train", one_row_per_impression=self.hparams.one_row_per_impression, **dataset_params
            )
            self.data_val: Optional[EbnerdDataset] = EbnerdDataset(
                mode="validation",
                user_id_to_index=self.data_train.user_id_to_index,
                article_id_to_index=self.data_train.article_id_to_index,
                train_df_behaviors=self.data_train.df_behaviors,
                one_row_per_impression=self.hparams.one_row_per_impression,
                **dataset_params
            )
            # one_row_per_impression should probably always be false for test
            self.data_test: Optional[EbnerdDataset] = EbnerdDataset(
                mode="test",
                user_id_to_index=self.data_val.user_id_to_index,
                article_id_to_index=self.data_val.article_id_to_index,
                train_df_behaviors=self.data_train.df_behaviors,
                one_row_per_impression=False,
                **dataset_params
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


