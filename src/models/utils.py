"""Model utilities."""

from typing import Any, Literal

from numpy import dtype, float_, ndarray, pad, str_

ArrayStr = ndarray[Any, dtype[str_]]

ArrayFloat = ndarray[Any, dtype[float_]]

Model = Literal["static", "contextual", "pooled"]

models: list[Model] = ["static", "contextual", "pooled"]


def padflat(embeddings: ndarray, window: int, dim: int) -> ndarray:
    """Flatten (n, d) embeddings and pad to (n * d,) where n = 2 * window + 1."""
    flat = embeddings.flatten()
    return pad(flat, (0, dim * (2 * window + 1) - len(flat)), constant_values=0)
