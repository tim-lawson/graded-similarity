"""Contextual embedding graded similarity estimator."""

from typing import TypedDict

from ..contextual_embedding import (
    BaseContextualEmbedding,
    ContextualEmbeddingName,
    get_contextual_embedding,
)
from ..similarity_measure import (
    BaseSimilarityMeasure,
    SimilarityMeasureName,
    get_similarity_measure,
)
from .base_graded_similarity import BaseGradedSimilarity


class ContextualGradedSimilarityParams(TypedDict):
    """Contextual embedding graded similarity estimator parameters."""

    contextual_embedding_name: ContextualEmbeddingName
    model_name: str
    similarity_measure_name: SimilarityMeasureName


class ContextualGradedSimilarity(BaseGradedSimilarity):
    """Contextual embedding graded similarity estimator."""

    _contextual: BaseContextualEmbedding | None = None
    _similarity_measure: BaseSimilarityMeasure | None = None

    def __init__(
        self,
        contextual_embedding_name: ContextualEmbeddingName,
        model_name: str,
        similarity_measure_name: SimilarityMeasureName,
    ):
        self.contextual_embedding_name = contextual_embedding_name
        self.model_name = model_name
        self.similarity_measure_name = similarity_measure_name

        self.set_params(
            contextual_embedding_name=contextual_embedding_name,
            model_name=model_name,
            similarity_measure_name=similarity_measure_name,
        )

    def set_params(self, **params):
        super().set_params(**params)

        self._contextual = get_contextual_embedding(
            self.contextual_embedding_name,
            self.model_name,
        )
        self._similarity_measure = get_similarity_measure(
            self.similarity_measure_name,
        )

        return self

    def predict(self, x):
        assert self._contextual is not None
        assert self._similarity_measure is not None
        return super().predict(x)
