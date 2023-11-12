"""Context window classes."""

from typing import Literal

from ..static_embedding import BaseStaticEmbedding
from .additive_context_window import AdditiveContextWindow
from .base_context_window import BaseContextWindow
from .mean_context_window import MeanContextWindow
from .multiplicative_context_window import MultiplicativeContextWindow

ContextWindowName = Literal["additive", "mean", "multiplicative"]


def get_context_window(
    static_embedding: BaseStaticEmbedding,
    context_window_name: ContextWindowName | str,
    context_window_size: int,
) -> BaseContextWindow:
    """Get the context window class."""

    if context_window_name == "additive":
        return AdditiveContextWindow(static_embedding, context_window_size)
    elif context_window_name == "mean":
        return MeanContextWindow(static_embedding, context_window_size)
    elif context_window_name == "multiplicative":
        return MultiplicativeContextWindow(static_embedding, context_window_size)

    raise ValueError(f"Unknown: {context_window_name}")
