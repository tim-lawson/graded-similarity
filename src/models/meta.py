"""Meta-model."""

from sklearn.base import BaseEstimator

from .contextual import PooledContextualBertModel, SimpleContextualBertModel
from .static import StaticBertModel
from .utils import Model


class MetaModel(BaseEstimator):
    """Meta-model."""

    def __init__(
        self,
        model: Model = "static",
        model_name: str = "bert-base-multilingual-cased",
        context_window_size: int = 0,
        context_window_operation: str = "none",
        similarity_measure: str = "cosine",
    ):
        self.model = model
        self.model_name = model_name
        self.context_window_size = context_window_size
        self.context_window_operation = context_window_operation
        self.similarity_measure = similarity_measure

    @property
    def _estimator(self):
        if self.model == "contextual":
            return SimpleContextualBertModel(
                self.model_name,
                self.context_window_size,
                self.context_window_operation,
                self.similarity_measure,
            )
        if self.model == "pooled":
            return PooledContextualBertModel(
                self.model_name,
                self.context_window_size,
                self.context_window_operation,
                self.similarity_measure,
            )
        if self.model == "static":
            return StaticBertModel(
                self.model_name,
                self.context_window_size,
                self.context_window_operation,
                self.similarity_measure,
            )
        raise ValueError(f"Unknown model: {self.model}")

    def fit(self, x, y):
        """Fit the model."""
        return self._estimator.fit(x, y)

    def predict(self, x):
        """Predict the change in similarity."""
        return self._estimator.predict(x)

    def score(self, x, y):
        """Compute the Pearson correlation coefficient."""
        return self._estimator.score(x, y)
