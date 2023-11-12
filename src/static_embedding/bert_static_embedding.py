"""BERT static embedding class."""

from typing import Literal

from transformers import (
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .base_static_embedding import BaseStaticEmbedding

BertModelName = Literal[
    "bert-base-uncased",
    "bert-base-multilingual-cased",
    "bert-large-uncased",
    "bert-large-uncased-whole-word-masking",
]


class BertStaticEmbedding(BaseStaticEmbedding):
    """BERT static embedding."""

    def __init__(self, model_name: str):
        self.model: PreTrainedModel = BertModel.from_pretrained(model_name)  # type: ignore
        self.tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(
            model_name
        )

    @property
    def embeddings(self):
        return self.model.get_input_embeddings().weight.detach().numpy()

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
