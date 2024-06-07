from enum import Enum
from dataclasses import dataclass


@dataclass
class ImageNetVariant:
    name: str
    path: str


class EbnerdVariants(Enum):
    EbnerdDemo = ImageNetVariant(
        name="ebnerd_demo",
        path="https://huggingface.co/datasets/glasswhiteboard/ebnerd/resolve/main/ebnerd_demo.zip?download=true"
    )

    EbnerdSmall = ImageNetVariant(
        name="ebnerd_small",
        path="https://ebnerd-dataset.s3.eu-west1.amazonaws.com/ebnerd_small.zip"
    )

    EbnerdLarge = ImageNetVariant(
        name="ebnerd_large",
        path="https://ebnerd-dataset.s3.eu-west1.amazonaws.com/ebnerd_large.zip"
    )

    EbnerdLargeArticles = ImageNetVariant(
        name="ebnerd_large_articles",
        path="https://ebnerd-dataset.s3.eu-west1.amazonaws.com/articles_large_only.zip"
    )


    @staticmethod
    def init_variant(variant: str):
        return {member.value.name: member for member in EbnerdVariants}[variant]

    @staticmethod
    def get_variants():
        return [member.value.name for member in EbnerdVariants]

    @staticmethod
    def get_default_variant():
        return EbnerdVariants.get_variants()[2]
