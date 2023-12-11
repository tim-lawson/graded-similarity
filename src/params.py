"""Model parameters."""

context_window_sizes = [
    1,
    2,
    5,
    10,
    20,
    50,
]

context_window_operations = [
    "none",
    "mean",
    "prod",
    "sum",
]

similarity_measures = [
    "cosine",
]

model_names = [
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-large-cased",
    "bert-large-cased-whole-word-masking",
]


def get_model_params():
    """Generate model parameters."""

    for model_name in model_names:
        for similarity_measure in similarity_measures:
            for context_window_operation in context_window_operations:
                if context_window_operation == "none":
                    yield {
                        "context_window_size": 0,
                        "context_window_operation": context_window_operation,
                        "similarity_measure": similarity_measure,
                        "model_name": model_name,
                    }
                else:
                    for context_window_size in context_window_sizes:
                        yield {
                            "context_window_size": context_window_size,
                            "context_window_operation": context_window_operation,
                            "similarity_measure": similarity_measure,
                            "model_name": model_name,
                        }


model_params_list = list(get_model_params())
