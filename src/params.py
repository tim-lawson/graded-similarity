"""Static and contextual graded similarity estimator parameters."""

from .context_window import ContextWindowName
from .contextual_embedding import BertModelName, ContextualEmbeddingName
from .graded_similarity import (
    ContextualGradedSimilarityParams,
    StaticGradedSimilarityParams,
)
from .similarity_measure import SimilarityMeasureName
from .static_embedding import StaticEmbeddingName

context_window_names: list[ContextWindowName] = [
    "additive",
    "mean",
    "multiplicative",
]

context_window_sizes: list[int] = [
    0,
    1,
    2,
    5,
    10,
    20,
    50,
]

static_embedding_names: list[StaticEmbeddingName] = [
    "bert",
]

contextual_embedding_names: list[ContextualEmbeddingName] = [
    "bert",
]

bert_model_names: list[BertModelName] = [
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-large-cased",
    "bert-large-cased-whole-word-masking",
]

similarity_measure_names: list[SimilarityMeasureName] = [
    "cosine",
]


def get_static_params():
    """Generate the static embedding graded similarity estimator parameters."""

    for context_window_name in context_window_names:
        for context_window_size in context_window_sizes:
            for similarity_measure_name in similarity_measure_names:
                for static_embedding_name in static_embedding_names:
                    if static_embedding_name == "bert":
                        for model_name in bert_model_names:
                            yield StaticGradedSimilarityParams(
                                context_window_name=context_window_name,
                                context_window_size=context_window_size,
                                similarity_measure_name=similarity_measure_name,
                                static_embedding_name=static_embedding_name,
                                model_name=model_name,
                            )


static_params = list(get_static_params())


def get_contextual_params():
    """Generate the contextual embedding graded similarity estimator parameters."""

    for similarity_measure_name in similarity_measure_names:
        for contextual_embedding_name in contextual_embedding_names:
            if contextual_embedding_name == "bert":
                for model_name in bert_model_names:
                    yield ContextualGradedSimilarityParams(
                        similarity_measure_name=similarity_measure_name,
                        contextual_embedding_name=contextual_embedding_name,
                        model_name=model_name,
                    )


contextual_params = list(get_contextual_params())
