"""Static embedding classes."""

from typing import Literal

from .base_static_embedding import BaseStaticEmbedding
from .bert_static_embedding import BertStaticEmbedding

StaticEmbeddingName = Literal["bert"]


def get_static_embedding(
    static_embedding_name: StaticEmbeddingName | str,
    model_name: str,
) -> BaseStaticEmbedding:
    """Get the static embedding class."""

    if static_embedding_name == "bert":
        return BertStaticEmbedding(model_name)
    raise ValueError(f"Unknown: {static_embedding_name}")
