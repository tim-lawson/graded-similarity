"""Similarity measure classes."""

from typing import Literal

from .base_similarity_measure import BaseSimilarityMeasure
from .cosine_similarity_measure import CosineSimilarityMeasure

SimilarityMeasureName = Literal["cosine"]


def get_similarity_measure(
    similarity_measure_name: SimilarityMeasureName | str,
) -> BaseSimilarityMeasure:
    """Get the similarity measure class."""

    if similarity_measure_name == "cosine":
        return CosineSimilarityMeasure()
    raise ValueError(f"Unknown similarity measure name: {similarity_measure_name}")
