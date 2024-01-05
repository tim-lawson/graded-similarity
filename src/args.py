"""Command-line arguments."""

from argparse import ArgumentParser
from typing import NamedTuple

from .data import Language, default_languages
from .models.utils import Embedding


class Args(NamedTuple):
    """Command-line arguments."""

    embedding: list[Embedding] = ["static", "contextual", "pooled"]
    model_name: list[str] = []
    language: list[Language] = default_languages
    window: list[int] | None = None
    min_window: int | None = None
    max_window: int | None = None
    operation: list[str] = ["none", "sum", "prod", "concat"]
    similarity: list[str] = ["cosine"]
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
            f"embedding = {','.join(self.embedding)}\n"
            f"model_name = {','.join(self.model_name)}\n"
            f"language = {','.join(self.language)}\n"
            f"window = {','.join(map(str, self.get_windows()))}\n"
            f"operation = {','.join(self.operation)}\n"
            f"similarity = {','.join(self.similarity)}"
        )

    def to_dict(self):
        """Convert to a dictionary."""
        return {
            "embedding": self.embedding,
            "model_name": self.model_name,
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
            # TODO: replace `model` by `embedding`, add `model_name`, rename CSVs
            f"model={'+'.join(self.embedding)}"
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
        "-e",
        "--embedding",
        nargs="+",
        default=["static", "contextual", "pooled"],
        help="embeddings",
    )

    parser.add_argument(
        "-m",
        "--model-name",
        nargs="+",
        default=[],
        help="model names",
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
        help="'practice kit'",
    )

    args = parser.parse_args()

    return Args(
        args.embedding,
        args.model_name,
        args.language,
        args.window,
        args.min_window,
        args.max_window,
        args.operation,
        args.similarity,
        args.practice,
    )
