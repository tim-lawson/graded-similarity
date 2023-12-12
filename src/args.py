"""Command-line arguments."""

from argparse import ArgumentParser
from typing import NamedTuple

from .data import Language, default_languages
from .models import Model


class Args(NamedTuple):
    """Command-line arguments."""

    model: list[Model]
    language: list[Language]
    window: list[int]
    operation: list[str]
    similarity: list[str]

    def __str__(self) -> str:
        return (
            f"model = {','.join(self.model)}\n"
            f"language = {','.join(self.language)}\n"
            f"window = {','.join(map(str, self.window))}\n"
            f"operation = {','.join(self.operation)}\n"
            f"similarity = {','.join(self.similarity)}"
        )

    def to_dict(self):
        """Convert to a dictionary."""
        return {
            "model": self.model,
            "language": self.language,
            "window": list(map(str, self.window)),
            "operation": self.operation,
            "similarity": self.similarity,
        }

    @property
    def filename(self) -> str:
        """Filename."""
        return (
            f"model={'+'.join(self.model)}"
            f"_language={'+'.join(self.language)}"
            f"_window={'+'.join(map(str, self.window))}"
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
        default=["static", "contextual"],
        help="static/contextual embeddings",
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
        "-w",
        "--window",
        nargs="+",
        type=int,
        default=[0, 1, 2, 5, 10, 20, 50, 100],
        help="context window sizes",
    )

    parser.add_argument(
        "-o",
        "--operation",
        nargs="+",
        type=str,
        default=["none", "sum"],
        help="context window operations",
    )

    parser.add_argument(
        "-s",
        "--similarity",
        nargs="+",
        type=str,
        default=["cosine"],
        help="similarity measures",
    )

    args = parser.parse_args()

    return Args(
        args.model,
        args.language,
        args.window,
        args.operation,
        args.similarity,
    )
