"""Base contextual embedding class."""

from abc import ABC, abstractmethod

from ..base_contextual import BaseContextual


class BaseContextualEmbedding(BaseContextual, ABC):
    """Base contextual embedding."""

    @abstractmethod
    def encode(self, text: str | list[str]) -> list[int]:
        """Encode text."""

    def find(self, word: str, context: str) -> int:
        """Find the index of the word in the context."""

        return self.encode(context).index(self.encode(word)[0])
