"""Command-line arguments."""

from argparse import ArgumentParser
from typing import NamedTuple

from .data import Language, default_languages
from .models.utils import Model


class Args(NamedTuple):
    """Command-line arguments."""

    model: list[Model]
    language: list[Language]
    window: list[int] | None
    min_window: int | None
    max_window: int | None
    operation: list[str]
    similarity: list[str]
    practice: bool = False

    def get_windows(self) -> list[int]:
        """Get the context window sizes."""
        if self.window is not None:
            return self.window
        if self.min_window is not None and self.max_window is not None:
            return list(range(self.min_window, self.max_window + 1))
        return []

    def __str__(self) -> str:
        return (
            f"model = {','.join(self.model)}\n"
            f"language = {','.join(self.language)}\n"
            f"window = {','.join(map(str, self.get_windows()))}\n"
            f"operation = {','.join(self.operation)}\n"
            f"similarity = {','.join(self.similarity)}"
        )

    def to_dict(self):
        """Convert to a dictionary."""
        return {
            "model": self.model,
            "language": self.language,
            "window": list(map(str, self.get_windows())),
            "operation": self.operation,
            "similarity": self.similarity,
        }

    @property
    def directory(self) -> str:
        """Directory."""
        return "results/practice" if self.practice else "results/evaluation"

    @property
    def filename(self) -> str:
        """Filename."""
        window = (
            f"_window={self.min_window}-{self.max_window}"
            if self.min_window is not None and self.max_window is not None
            else f"_window={'+'.join(map(str, self.get_windows()))}"
        )
        return (
            f"model={'+'.join(self.model)}"
            f"_language={'+'.join(self.language)}"
            f"{window}"
            f"_operation={'+'.join(self.operation)}"
            f"_similarity={'+'.join(self.similarity)}"
            ".csv"
        )


def parse_args() -> Args:
    """Command-line arguments."""

    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        default=["static", "contextual", "pooled"],
        help="embeddings",
    )

    parser.add_argument(
        "-l",
        "--language",
        nargs="+",
        type=str,
        default=default_languages,
        help="languages",
    )

    parser.add_argument(
        "-min",
        "--min-window",
        type=int,
        help="minimum context-window size",
    )

    parser.add_argument(
        "-max",
        "--max-window",
        type=int,
        help="maximum context-window size",
    )

    parser.add_argument(
        "-w",
        "--window",
        nargs="+",
        type=int,
        help="context-window sizes",
    )

    parser.add_argument(
        "-o",
        "--operation",
        nargs="+",
        type=str,
        default=["none", "sum", "prod", "concat"],
        help="context-window operations",
    )

    parser.add_argument(
        "-s",
        "--similarity",
        nargs="+",
        type=str,
        default=["cosine"],
        help="similarity measures",
    )

    parser.add_argument(
        "-p",
        "--practice",
        action="store_true",
        help="practice kit",
    )

    args = parser.parse_args()

    return Args(
        args.model,
        args.language,
        args.window,
        args.min_window,
        args.max_window,
        args.operation,
        args.similarity,
        args.practice,
    )
