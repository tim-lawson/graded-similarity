"""Base similarity measure class."""

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class BaseSimilarityMeasure(ABC):
    """Base similarity measure."""

    @abstractmethod
    def similarity(self, embedding1: ArrayLike, embedding2: ArrayLike) -> float:
        """Compute the similarity between two embeddings."""

    def change(
        self,
        embeddings1: tuple[ArrayLike, ArrayLike],
        embeddings2: tuple[ArrayLike, ArrayLike],
    ) -> float:
        """Compute the change in similarity between two pairs of embeddings."""

        return self.similarity(*embeddings2) - self.similarity(*embeddings1)
