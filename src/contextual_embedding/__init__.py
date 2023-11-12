"""Contextual embedding classes."""

from typing import Literal

from .base_contextual_embedding import BaseContextualEmbedding
from .bert_contextual_embedding import BertContextualEmbedding, BertModelName

ContextualEmbeddingName = Literal["bert"]


def get_contextual_embedding(
    contextual_embedding_name: ContextualEmbeddingName | str,
    model_name: BertModelName | str,
) -> BaseContextualEmbedding:
    """Get the contextual embedding class."""

    if contextual_embedding_name == "bert":
        return BertContextualEmbedding(model_name)

    raise ValueError(f"Unknown: {contextual_embedding_name}")
