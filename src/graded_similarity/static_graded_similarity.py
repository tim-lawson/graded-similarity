"""Static embedding graded similarity estimator."""

from typing import TypedDict

from ..context_window import BaseContextWindow, ContextWindowName, get_context_window
from ..similarity_measure import (
    BaseSimilarityMeasure,
    SimilarityMeasureName,
    get_similarity_measure,
)
from ..static_embedding import (
    BaseStaticEmbedding,
    StaticEmbeddingName,
    get_static_embedding,
)
from .base_graded_similarity import BaseGradedSimilarity


class StaticGradedSimilarityParams(TypedDict):
    """Static embedding graded similarity estimator parameters."""

    static_embedding_name: StaticEmbeddingName
    model_name: str
    context_window_name: ContextWindowName
    context_window_size: int
    similarity_measure_name: SimilarityMeasureName


class StaticGradedSimilarity(BaseGradedSimilarity):
    """Static embedding graded similarity estimator."""

    _static_embedding: BaseStaticEmbedding | None = None
    _contextual: BaseContextWindow | None = None
    _similarity_measure: BaseSimilarityMeasure | None = None

    def __init__(
        self,
        static_embedding_name: StaticEmbeddingName,
        model_name: str,
        context_window_name: ContextWindowName,
        context_window_size: int,
        similarity_measure_name: SimilarityMeasureName,
    ):
        self.static_embedding_name = static_embedding_name
        self.model_name = model_name
        self.context_window_name = context_window_name
        self.context_window_size = context_window_size
        self.similarity_measure_name = similarity_measure_name

        self.set_params(
            static_embedding_name=static_embedding_name,
            model_name=model_name,
            context_window_name=context_window_name,
            context_window_size=context_window_size,
            similarity_measure_name=similarity_measure_name,
        )

    def set_params(self, **params):
        super().set_params(**params)

        self._static_embedding = get_static_embedding(
            self.static_embedding_name,
            self.model_name,
        )
        self._contextual = get_context_window(
            self._static_embedding,
            self.context_window_name,
            self.context_window_size,
        )
        self._similarity_measure = get_similarity_measure(
            self.similarity_measure_name,
        )

        return self

    def predict(self, x):
        assert self._static_embedding is not None
        assert self._contextual is not None
        assert self._similarity_measure is not None
        return super().predict(x)
