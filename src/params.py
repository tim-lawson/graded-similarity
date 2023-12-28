"""Model parameters."""

from typing import NamedTuple

from .data import Language
from .models.utils import Embedding

model_names_multilingual = [
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "EMBEDDIA/crosloengual-bert",
]


def get_model_names(language: Language) -> list[str]:
    """Get language-specific BERT model names."""

    if language == "en":
        return model_names_multilingual + [
            "bert-base-cased",
            "bert-base-uncased",
            "bert-large-cased",
            "bert-large-uncased",
            "bert-large-cased-whole-word-masking",
            "bert-large-uncased-whole-word-masking",
        ]

    if language == "fi":
        return model_names_multilingual + [
            "TurkuNLP/bert-base-finnish-cased-v1",
            "TurkuNLP/bert-base-finnish-uncased-v1",
            "TurkuNLP/bert-large-finnish-cased-v1",
        ]

    if language == "hr":
        return model_names_multilingual + [
            "classla/bcms-bertic",
        ]

    if language == "sl":
        return model_names_multilingual

    return []


class Params(NamedTuple):
    """Model parameters."""

    language: str
    embedding: Embedding
    model_name: str
    window: int
    operation: str
    similarity: str

    def __str__(self) -> str:
        return (
            f"embedding = {self.embedding}\n"
            f"model_name = {self.model_name}\n"
            f"language = {self.language}\n"
            f"window = {self.window}\n"
            f"operation = {self.operation}\n"
            f"similarity = {self.similarity}"
        )

    def to_dict(self) -> dict[str, str]:
        """Convert to a dictionary."""
        return {
            "embedding": self.embedding,
            "model_name": self.model_name,
            "language": self.language,
            "window": str(self.window),
            "operation": self.operation,
            "similarity": self.similarity,
        }

    @property
    def filename(self) -> str:
        """Filename."""
        return (
            f"embedding={self.embedding}"
            f"_model_name={self.model_name.replace('/', '-')}"
            f"_language={self.language}"
            f"_window={self.window}"
            f"_operation={self.operation}"
            f"_similarity={self.similarity}"
            ".csv"
        )
