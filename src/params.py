"""Model parameters."""

from .data import Language

context_window_operations = {
    "none": [0],
    "mean": [1, 2, 5, 10, 20, 50],
    "prod": [1, 2, 5],
    "sum": [1, 2, 5, 10, 20, 50],
}

similarity_measures = [
    "cosine",
]

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
        return model_names_multilingual + [
            "gerulata/slovakbert",
        ]

    return []


def get_params(language: Language):
    """Get language-specific model parameters."""

    params = []

    for model_name in get_model_names(language):
        for similarity_measure in similarity_measures:
            for (
                context_window_operation,
                context_window_sizes,
            ) in context_window_operations.items():
                for context_window_size in context_window_sizes:
                    params.append(
                        {
                            "context_window_size": context_window_size,
                            "context_window_operation": context_window_operation,
                            "similarity_measure": similarity_measure,
                            "model_name": model_name,
                        }
                    )

    return params
