"""Base graded similarity estimator."""

from typing import Any

from numpy import apply_along_axis, array, dtype, float_, ndarray, str_
from scipy.spatial import distance
from sklearn.base import BaseEstimator

from ..base_contextual import BaseContextual
from ..similarity_measure import BaseSimilarityMeasure

ArrayStr = ndarray[Any, dtype[str_]]
ArrayFloat = ndarray[Any, dtype[float_]]


class BaseGradedSimilarity(BaseEstimator):
    """Base graded similarity estimator."""

    _contextual: BaseContextual
    _similarity_measure: BaseSimilarityMeasure

    def __init__(
        self,
    ):
        pass

    def fit(self, _x, _y):
        """No-op."""
        return self

    def predict(self, x: ArrayStr) -> ArrayFloat:
        """Predict the change in similarity."""

        def similarity(row: ArrayStr):
            word1_con1 = self._contextual.embedding(*row[[0, 2, 4]])
            word2_con1 = self._contextual.embedding(*row[[1, 2, 5]])
            word1_con2 = self._contextual.embedding(*row[[0, 3, 6]])
            word2_con2 = self._contextual.embedding(*row[[1, 3, 7]])
            sim_con1 = self._similarity_measure.similarity(word1_con1, word2_con1)
            sim_con2 = self._similarity_measure.similarity(word1_con2, word2_con2)
            return sim_con2 - sim_con1

        return apply_along_axis(similarity, 1, array(x, dtype=str_))

    def score(self, x: ArrayStr, y: ArrayFloat):
        """
        Compute the Pearson correlation coefficient between the predicted changes in
        similarity and the gold-standard values.
        """

        return 1.0 - distance.correlation(self.predict(x), y, centered=False)
