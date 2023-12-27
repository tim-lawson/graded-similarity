"""Static-embedding model."""

from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .base import BaseModel
from .utils import ArrayFloat


class StaticBertModel(BaseModel):
    """BERT static-embedding model."""

    def __init__(
        self,
        model_name: str,
        context_window_size: int,
        context_window_operation: str,
        similarity_measure: str,
    ):
        super().__init__(
            model_name,
            context_window_size,
            context_window_operation,
            similarity_measure,
        )

        self.model: PreTrainedModel
        self.tokenizer: PreTrainedTokenizer
        self._set_model()

    def set_params(self, **params) -> None:
        super().set_params(**params)
        self._set_model()

    def _set_model(self):
        if self.model_name in ["classla/bcms-bertic"]:
            self.model = ElectraModel.from_pretrained(
                self.model_name,
                config=ElectraConfig.from_pretrained(
                    self.model_name, output_hidden_states=True
                ),
            )  # type: ignore
            self.tokenizer = ElectraTokenizer.from_pretrained(self.model_name)

        else:
            self.model = BertModel.from_pretrained(
                self.model_name,
                config=BertConfig.from_pretrained(
                    self.model_name, output_hidden_states=True
                ),
            )  # type: ignore
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @property
    def _static_embeddings(self) -> ArrayFloat:
        return self.model.get_input_embeddings().weight.detach().numpy()

    def _embeddings(self, _context: str) -> ArrayFloat:
        return self._static_embeddings

    def _embedding(self, word: str, context: str, word_context: str) -> ArrayFloat:
        tokens = self._encode(context)

        if self.context_window_operation == "none" or self.context_window_size == 0:
            return self._static_embeddings[tokens[self._find(word_context, context)]]

        start, end = self._context_window(word, context, word_context)
        return self._compose(self._embeddings(context)[tokens[start:end]])
