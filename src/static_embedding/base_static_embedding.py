"""Base static embedding class."""

from abc import ABC, abstractmethod

from numpy import ndarray


class BaseStaticEmbedding(ABC):
    """Base static embedding."""

    @property
    @abstractmethod
    def embeddings(self) -> ndarray:
        """The static embeddings."""

    @abstractmethod
    def encode(self, text: str | list[str]) -> list[int]:
        """Encode text."""

    def find(self, word: str, context: str) -> int:
        """Find the index of the word in the context."""

        return self.encode(context).index(self.encode(word)[0])
