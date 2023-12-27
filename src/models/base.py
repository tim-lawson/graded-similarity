"""Base model."""

from numpy import apply_along_axis, array, str_
from scipy.spatial.distance import correlation, cosine
from sklearn.base import BaseEstimator

from .utils import ArrayFloat, ArrayStr, padflat


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
        if self.context_window_operation == "concat":
            return padflat(embeddings, self.context_window_size, embeddings.shape[1])
        if self.context_window_operation == "mean":
            return embeddings.mean(axis=0)
        if self.context_window_operation == "none":
            return embeddings.flatten()
        if self.context_window_operation == "prod":
            return embeddings.prod(axis=0)
        if self.context_window_operation == "sum":
            return embeddings.sum(axis=0)
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
