"""A script to get the best results and generate additional data for the report."""

# pylint: disable=protected-access

import os

from numpy import average
from pandas import DataFrame, concat, read_csv

from .data import Language, default_languages, load_x
from .models.static import StaticBertModel

columns = [
    "model",
    "model_name",
    "language",
    "window",
    "operation",
    "similarity",
    "score",
    "time",
]


def _get_top_1_model_name(filename: str):
    read_csv(filename, header=0, names=columns).sort_values(
        by=["score"], ascending=False
    ).groupby("model_name").head(1).reset_index(drop=True).sort_values(
        by=["model_name"]
    ).to_csv(
        filename.replace(".csv", "_top_1.csv"), index=False
    )


ns: dict[Language, int] = {
    "en": 340,
    "fi": 24,
    "hr": 112,
    "sl": 111,
}


def _get_time_per_instance(filename: str, n: int):
    dataframe = read_csv(filename, header=0, names=columns)
    dataframe["time"] = dataframe["time"] / n
    dataframe.to_csv(filename.replace(".csv", "_time.csv"), index=False)


def _get_best(practice: bool = False):
    results_directory = "practice" if practice else "evaluation"

    dataframes: list[DataFrame] = []
    for filename in os.listdir(f"results/{results_directory}"):
        if filename.endswith(".csv"):
            dataframes.append(
                read_csv(
                    f"results/{results_directory}/{filename}",
                    header=0,
                    names=columns,
                )
            )

    dataframe = concat(dataframes).sort_values(by=["score"], ascending=False)

    dataframe.groupby(["language"]).head(1).reset_index(drop=True).sort_values(
        by=["language"]
    ).to_csv(f"results/best/{results_directory}_best_overall.csv", index=False)

    for model in ["static", "contextual", "pooled"]:
        (
            dataframe[dataframe["model"] == model]
            .groupby(["language"])
            .head(1)
            .reset_index(drop=True)
            .sort_values(by=["language"])
            .to_csv(f"results/best/{results_directory}_best_{model}.csv", index=False)
        )


def _get_filename(
    language: Language,
    model: str,
    min_window: int,
    max_window: int,
    operation: str,
    similarity: str,
):
    return (
        f"results/evaluation/model={model}_"
        f"language={language}_"
        f"window={min_window}-{max_window}_"
        f"operation={operation}_"
        f"similarity={similarity}.csv"
    )


def _get_results():
    for language in default_languages:
        for filename in [
            _get_filename(language, "static", 0, 50, "sum", "cosine"),
            _get_filename(language, "static", 0, 50, "concat", "cosine"),
            _get_filename(language, "contextual", 0, 10, "sum", "cosine"),
            _get_filename(language, "contextual", 0, 10, "concat", "cosine"),
            _get_filename(language, "pooled", 0, 10, "sum", "cosine"),
        ]:
            _get_top_1_model_name(filename)
            _get_time_per_instance(filename, ns[language])


def _get_token_examples():
    row = 2

    for language in default_languages:
        print(f"language = {language}")

        x = load_x(language).to_numpy()

        for window in [0, 1, 2, 3]:
            model = StaticBertModel(
                "bert-base-multilingual-uncased", window, "sum", "cosine"
            )

            context = model._encode(x[row][2])

            start, end = model._context_window(x[row][0], x[row][2], x[row][4])

            tokens = context[start:end]

            print(
                f"window = {window}, "
                f"words = '{model._decode(tokens)}', "
                f"subwords = '{' '.join([model._decode([token]) for token in tokens])}'"
            )


def _get_context_lengths():
    for language in default_languages:
        print(f"language = {language}")

        x = load_x(language).to_numpy()

        print(
            f"average context length = {average([len(row[2].split(' ')) for row in x])}"
        )


if __name__ == "__main__":
    _get_best()
    # _get_best(True)
    _get_context_lengths()
    _get_results()
    _get_token_examples()
