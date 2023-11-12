"""Additive context window class."""

from ..base_contextual import BaseContextual
from .base_context_window import BaseContextWindow


class AdditiveContextWindow(BaseContextual, BaseContextWindow):
    """Additive context window."""

    def embedding(self, word, context, word_context):
        return self.static_embedding.embeddings[
            self.window(word, context, word_context)
        ].sum(axis=0)
