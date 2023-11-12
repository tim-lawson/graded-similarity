"""BERT contextual embedding class."""

from typing import Literal

from numpy import ndarray
from transformers import (
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .base_contextual_embedding import BaseContextualEmbedding

BertModelName = Literal[
    "bert-base-cased",
    "bert-base-uncased",
    "bert-base-multilingual-cased",
    "bert-large-cased",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased",
    "bert-large-uncased-whole-word-masking",
]


class BertContextualEmbedding(BaseContextualEmbedding):
    """BERT contextual embeddings."""

    def __init__(self, model_name: str):
        self.model: PreTrainedModel = BertModel.from_pretrained(model_name)  # type: ignore
        self.tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(
            model_name
        )

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def embedding(self, _word: str, context: str, word_context: str) -> ndarray:
        word_index = self.find(word_context, context)

        outputs = self.model(**self.tokenizer(context, return_tensors="pt"))

        embeddings = outputs[0][0]

        return embeddings[word_index].detach().numpy()
