"""Base context window class."""

from ..static_embedding import BaseStaticEmbedding


class BaseContextWindow:
    """Base context window."""

    def __init__(self, static_embedding: BaseStaticEmbedding, window_size: int):
        self.static_embedding = static_embedding
        self.window_size = window_size

    def window(self, _word: str, context: str, word_context: str) -> list[int]:
        """The encoded context window."""

        context_encoded = self.static_embedding.encode(context)

        word_index = self.static_embedding.find(word_context, context)

        return context_encoded[
            max(0, word_index - self.window_size) : word_index + self.window_size + 1
        ]
