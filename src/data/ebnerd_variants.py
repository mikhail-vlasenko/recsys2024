from enum import Enum
from dataclasses import dataclass


@dataclass
class ImageNetVariant:
    name: str
    path: str


class EbnerdVariants(Enum):
    EbnerdDemo = ImageNetVariant(
        name="ebnerd_demo",
        path="https://huggingface.co/datasets/recsys2024/ebnerd/resolve/main/ebnerd_demo.zip?download=true"
    )

    EbnerdSmall = ImageNetVariant(
        name="ebnerd_small",
        path="https://huggingface.co/datasets/recsys2024/ebnerd/resolve/main/ebnerd_small.zip?download=true"
    )

    EbnerdLarge = ImageNetVariant(
        name="ebnerd_large",
        path="https://huggingface.co/datasets/recsys2024/ebnerd/resolve/main/ebnerd_large.zip?download=true"
    )

    EbnerdLargeArticles = ImageNetVariant(
        name="ebnerd_large_articles",
        path="https://huggingface.co/datasets/recsys2024/ebnerd/resolve/main/articles_large_only.zip?download=true"
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
