"""Cosine similarity measure class."""

from scipy.spatial import distance

from .base_similarity_measure import BaseSimilarityMeasure


class CosineSimilarityMeasure(BaseSimilarityMeasure):
    """Cosine similarity measure."""

    def similarity(self, embedding1, embedding2):
        return 1.0 - distance.cosine(embedding1, embedding2)
