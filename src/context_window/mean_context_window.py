"""Mean context window class."""

from ..base_contextual import BaseContextual
from .base_context_window import BaseContextWindow


class MeanContextWindow(BaseContextual, BaseContextWindow):
    """Mean context window."""

    def embedding(self, word, context, word_context):
        return self.static_embedding.embeddings[
            self.window(word, context, word_context)
        ].mean(axis=0)
