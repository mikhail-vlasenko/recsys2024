from src.utils import get_training_args
from src.data.original_model_datamodule import OriginalModelDatamodule
from src.data.ebnerd_variants import EbnerdVariants
import wandb

def main():
    args = get_training_args()

    data_download_path = EbnerdVariants.init_variant(args.ebnerd_variant).path

    datamodule = OriginalModelDatamodule(data_download_path=data_download_path, batch_size=args.batch_size, num_workers=args.num_workers)

    wandb.login()


if __name__ == "__main__":
    main()