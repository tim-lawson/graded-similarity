"""Base contextualised embedding class."""

from abc import ABC, abstractmethod

from numpy import ndarray


class BaseContextual(ABC):
    """Base contextualised embedding."""

    @abstractmethod
    def embedding(self, word: str, context: str, word_context: str) -> ndarray:
        """The embedding of the word in the context."""
