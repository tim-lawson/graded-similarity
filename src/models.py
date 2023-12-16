"""Static- and contextual-embedding BERT models."""

from typing import Any, Literal, Sequence

from numpy import apply_along_axis, array, average, dtype, float_, ndarray, pad, str_
from scipy.spatial.distance import correlation, cosine
from sklearn.base import BaseEstimator
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
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

ArrayStr = ndarray[Any, dtype[str_]]

ArrayFloat = ndarray[Any, dtype[float_]]


def padflat(embeddings: ndarray, window: int, dim: int) -> ndarray:
    """Flatten (n, d) embeddings and pad to (n * d,) where n = 2 * window + 1."""
    flat = embeddings.flatten()
    return pad(flat, (0, dim * (2 * window + 1) - len(flat)), constant_values=0)


class BaseModel(BaseEstimator):
    """Base model."""

    def __init__(
        self,
        model_name: str,
        context_window_size: int,
        context_window_operation: str,
        similarity_measure: str,
    ):
        self.model_name = model_name
        self.context_window_size = context_window_size
        self.context_window_operation = context_window_operation
        self.similarity_measure = similarity_measure

    def _encode(self, text: str | list[str]) -> list[int]:
        raise NotImplementedError

    def _decode(self, tokens: list[int]) -> list[str]:
        raise NotImplementedError

    def _find(self, word: str, context: str) -> int:
        return self._encode(context).index(self._encode(word)[0])

    def _context_window(
        self, _word: str, context: str, word_context: str
    ) -> tuple[int, int]:
        index = self._find(word_context, context)
        return (
            max(0, index - self.context_window_size),
            index + self.context_window_size + 1,
        )

    def _compose(
        self,
        embeddings: ArrayFloat,
    ) -> ArrayFloat:
        if self.context_window_operation == "mean":
            return embeddings.mean(axis=0)
        if self.context_window_operation == "prod":
            return embeddings.prod(axis=0)
        if self.context_window_operation == "sum":
            return embeddings.sum(axis=0)
        if self.context_window_operation == "concat":
            return padflat(embeddings, self.context_window_size, embeddings.shape[1])
        if self.context_window_operation == "none":
            return embeddings.flatten()
        raise ValueError(
            f"Unknown context window operation: {self.context_window_operation}"
        )

    def _embedding(self, _word: str, _context: str, _word_context: str) -> ArrayFloat:
        raise NotImplementedError

    def _similarity(
        self, word1_context: ArrayFloat, word2_context: ArrayFloat
    ) -> float:
        if self.similarity_measure == "cosine":
            return 1.0 - float(cosine(word1_context, word2_context))
        raise ValueError(f"Unknown similarity measure: {self.similarity_measure}")

    def _change(
        self,
        word1_context1: ArrayFloat,
        word2_context1: ArrayFloat,
        word1_context2: ArrayFloat,
        word2_context2: ArrayFloat,
    ) -> float:
        sim_context1 = self._similarity(word1_context1, word2_context1)
        sim_context2 = self._similarity(word1_context2, word2_context2)
        return sim_context2 - sim_context1

    def fit(self, _x, _y):
        """No-op."""
        return self

    def predict(self, x: ArrayStr) -> ArrayFloat:
        """Predict the change in similarity."""

        def change(row: ArrayStr) -> float:
            return self._change(
                self._embedding(*row[[0, 2, 4]]),
                self._embedding(*row[[1, 2, 5]]),
                self._embedding(*row[[0, 3, 6]]),
                self._embedding(*row[[1, 3, 7]]),
            )

        predictions = apply_along_axis(change, 1, array(x, dtype=str_))
        return predictions

    def score(self, x: ArrayStr, y: ArrayFloat):
        """Compute the Pearson correlation coefficient."""
        return 1.0 - float(correlation(self.predict(x), y, centered=False))


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


class SimpleContextualBertModel(StaticBertModel):
    """BERT contextual-embedding model (last hidden-state)."""

    def _embeddings(self, context: str) -> ArrayFloat:
        return (
            self.model(**self.tokenizer(context, return_tensors="pt"))[0][0]
            .detach()
            .numpy()
        )

    def _embedding(self, word: str, context: str, word_context: str) -> ArrayFloat:
        if self.context_window_operation == "none" or self.context_window_size == 0:
            return self._embeddings(context)[self._find(word_context, context)]

        start, end = self._context_window(word, context, word_context)
        return self._compose(self._embeddings(context)[start:end])


class PooledContextualBertModel(StaticBertModel):
    """BERT contextual-embedding model (last four hidden-states)."""

    def _embeddings(self, context: str) -> ArrayFloat:
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.model(
            **self.tokenizer(context, return_tensors="pt")
        )

        assert outputs.hidden_states is not None

        # Get the last four hidden-states
        hidden_states0 = outputs.hidden_states[-1][0].detach().numpy()
        hidden_states1 = outputs.hidden_states[-2][0].detach().numpy()  # type: ignore
        hidden_states2 = outputs.hidden_states[-3][0].detach().numpy()  # type: ignore
        hidden_states3 = outputs.hidden_states[-4][0].detach().numpy()  # type: ignore

        # Get the average of the hidden-states
        hidden_states = (
            hidden_states0 + hidden_states1 + hidden_states2 + hidden_states3
        )

        return hidden_states

    def _embedding(self, word: str, context: str, word_context: str) -> ArrayFloat:
        if self.context_window_operation == "none" or self.context_window_size == 0:
            return self._embeddings(context)[self._find(word_context, context)]

        start, end = self._context_window(word, context, word_context)
        return self._compose(self._embeddings(context)[start:end])


class EnsembleModel(BaseEstimator):
    """Ensemble model."""

    def __init__(
        self,
        models: Sequence[BaseModel],
        weights: bool = False,
    ):
        self.models = models
        self.weights = weights

    def fit(self, _x, _y):
        """No-op."""
        return self

    def predict(self, x: ArrayStr, y: ArrayFloat) -> ArrayFloat:
        """Predict the change in similarity."""
        if self.weights:
            weights = array([model.score(x, y) for model in self.models])
        else:
            weights = None
        predictions = array([model.predict(x) for model in self.models])
        return average(predictions, axis=0, weights=weights)

    def score(self, x: ArrayStr, y: ArrayFloat):
        """Compute the Pearson correlation coefficient."""
        return 1.0 - float(correlation(self.predict(x, y), y, centered=False))


Model = Literal["static", "contextual", "pooled"]

model_types: dict[Model, type[BaseModel]] = {
    "static": StaticBertModel,
    "contextual": SimpleContextualBertModel,
    "pooled": PooledContextualBertModel,
}
