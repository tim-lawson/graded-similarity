"""Model ensembles."""

from .data import Language
from .models import (
    PooledContextualBertModel,
    SimpleContextualBertModel,
    StaticBertModel,
)


def get_static_ensemble(language: Language):
    """Get an ensemble of static models."""
    if language == "en":
        return [
            StaticBertModel("EMBEDDIA/crosloengual-bert", 17, "sum", "cosine"),
            StaticBertModel("bert-base-cased", 18, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-cased", 18, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-uncased", 19, "sum", "cosine"),
            StaticBertModel("bert-base-uncased", 14, "sum", "cosine"),
            StaticBertModel("bert-large-cased", 18, "sum", "cosine"),
            StaticBertModel("bert-large-cased-whole-word-masking", 18, "sum", "cosine"),
            StaticBertModel("bert-large-uncased", 16, "sum", "cosine"),
            StaticBertModel(
                "bert-large-uncased-whole-word-masking", 16, "sum", "cosine"
            ),
        ]
    if language == "fi":
        return [
            StaticBertModel("EMBEDDIA/crosloengual-bert", 21, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-cased", 3, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-uncased", 29, "sum", "cosine"),
            StaticBertModel("TurkuNLP/bert-base-finnish-cased-v1", 0, "sum", "cosine"),
            StaticBertModel(
                "TurkuNLP/bert-base-finnish-uncased-v1", 0, "sum", "cosine"
            ),
            StaticBertModel("TurkuNLP/bert-large-finnish-cased-v1", 0, "sum", "cosine"),
        ]
    if language == "hr":
        return [
            StaticBertModel("EMBEDDIA/crosloengual-bert", 31, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-cased", 32, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-uncased", 31, "sum", "cosine"),
            StaticBertModel("classla/bcms-bertic", 31, "sum", "cosine"),
        ]
    if language == "sl":
        return [
            StaticBertModel("EMBEDDIA/crosloengual-bert", 11, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-cased", 8, "sum", "cosine"),
            StaticBertModel("bert-base-multilingual-uncased", 10, "sum", "cosine"),
        ]

    raise ValueError(f"Unknown language: {language}")


def get_contextual_ensemble(language: Language):
    """Get an ensemble of contextual models."""
    if language == "en":
        return [
            SimpleContextualBertModel("EMBEDDIA/crosloengual-bert", 3, "sum", "cosine"),
            SimpleContextualBertModel("bert-base-cased", 1, "sum", "cosine"),
            SimpleContextualBertModel(
                "bert-base-multilingual-cased", 1, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "bert-base-multilingual-uncased", 1, "sum", "cosine"
            ),
            SimpleContextualBertModel("bert-base-uncased", 3, "sum", "cosine"),
            SimpleContextualBertModel("bert-large-cased", 3, "sum", "cosine"),
            SimpleContextualBertModel(
                "bert-large-cased-whole-word-masking", 2, "sum", "cosine"
            ),
            SimpleContextualBertModel("bert-large-uncased", 1, "sum", "cosine"),
            SimpleContextualBertModel(
                "bert-large-uncased-whole-word-masking", 1, "sum", "cosine"
            ),
        ]
    if language == "fi":
        return [
            SimpleContextualBertModel(
                "EMBEDDIA/crosloengual-bert", 10, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "bert-base-multilingual-cased", 3, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "bert-base-multilingual-uncased", 2, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "TurkuNLP/bert-base-finnish-cased-v1", 1, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "TurkuNLP/bert-base-finnish-uncased-v1", 1, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "TurkuNLP/bert-large-finnish-cased-v1", 1, "sum", "cosine"
            ),
        ]
    if language == "hr":
        return [
            SimpleContextualBertModel("EMBEDDIA/crosloengual-bert", 3, "sum", "cosine"),
            SimpleContextualBertModel(
                "bert-base-multilingual-cased", 5, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "bert-base-multilingual-uncased", 6, "sum", "cosine"
            ),
            SimpleContextualBertModel("classla/bcms-bertic", 10, "sum", "cosine"),
        ]
    if language == "sl":
        return [
            SimpleContextualBertModel("EMBEDDIA/crosloengual-bert", 3, "sum", "cosine"),
            SimpleContextualBertModel(
                "bert-base-multilingual-cased", 3, "sum", "cosine"
            ),
            SimpleContextualBertModel(
                "bert-base-multilingual-uncased", 9, "sum", "cosine"
            ),
        ]

    raise ValueError(f"Unknown language: {language}")


def get_pooled_ensemble(language: Language):
    """Get an ensemble of pooled models."""
    if language == "en":
        return [
            PooledContextualBertModel("EMBEDDIA/crosloengual-bert", 1, "sum", "cosine"),
            PooledContextualBertModel("bert-base-cased", 1, "sum", "cosine"),
            PooledContextualBertModel(
                "bert-base-multilingual-cased", 1, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "bert-base-multilingual-uncased", 1, "sum", "cosine"
            ),
            PooledContextualBertModel("bert-base-uncased", 1, "sum", "cosine"),
            PooledContextualBertModel("bert-large-cased", 1, "sum", "cosine"),
            PooledContextualBertModel(
                "bert-large-cased-whole-word-masking", 3, "sum", "cosine"
            ),
            PooledContextualBertModel("bert-large-uncased", 3, "sum", "cosine"),
            PooledContextualBertModel(
                "bert-large-uncased-whole-word-masking", 1, "sum", "cosine"
            ),
        ]
    if language == "fi":
        return [
            PooledContextualBertModel(
                "EMBEDDIA/crosloengual-bert", 10, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "bert-base-multilingual-cased", 1, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "bert-base-multilingual-uncased", 1, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "TurkuNLP/bert-base-finnish-cased-v1", 1, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "TurkuNLP/bert-base-finnish-uncased-v1", 2, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "TurkuNLP/bert-large-finnish-cased-v1", 2, "sum", "cosine"
            ),
        ]
    if language == "hr":
        return [
            PooledContextualBertModel("EMBEDDIA/crosloengual-bert", 3, "sum", "cosine"),
            PooledContextualBertModel(
                "bert-base-multilingual-cased", 6, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "bert-base-multilingual-uncased", 1, "sum", "cosine"
            ),
            PooledContextualBertModel("classla/bcms-bertic", 2, "sum", "cosine"),
        ]
    if language == "sl":
        return [
            PooledContextualBertModel("EMBEDDIA/crosloengual-bert", 2, "sum", "cosine"),
            PooledContextualBertModel(
                "bert-base-multilingual-cased", 3, "sum", "cosine"
            ),
            PooledContextualBertModel(
                "bert-base-multilingual-uncased", 2, "sum", "cosine"
            ),
        ]

    raise ValueError(f"Unknown language: {language}")
