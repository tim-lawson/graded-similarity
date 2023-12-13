"""Utilities to convert the results and generate data for the report."""

# pylint: disable=protected-access

from numpy import average
from pandas import read_csv

from .data import default_languages, load_x
from .models import StaticBertModel

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


def _top_1_model_name(filename: str):
    read_csv(filename, header=0, names=columns).sort_values(
        by=["score"], ascending=False
    ).groupby("model_name").head(1).reset_index(drop=True).sort_values(
        by=["model_name"]
    ).to_csv(
        filename.replace(".csv", "_top_1.csv"), index=False
    )


def _results():
    for filename in [
        "results/model=contextual_language=en_window=0-10_operation=sum_similarity=cosine.csv",
        "results/model=contextual_language=fi_window=0-10_operation=sum_similarity=cosine.csv",
        "results/model=contextual_language=hr_window=0-10_operation=sum_similarity=cosine.csv",
        "results/model=contextual_language=sl_window=0-10_operation=sum_similarity=cosine.csv",
        "results/model=static_language=en_window=0-50_operation=sum_similarity=cosine.csv",
        "results/model=static_language=fi_window=0-50_operation=sum_similarity=cosine.csv",
        "results/model=static_language=hr_window=0-50_operation=sum_similarity=cosine.csv",
        "results/model=static_language=sl_window=0-50_operation=sum_similarity=cosine.csv",
    ]:
        _top_1_model_name(filename)


def _examples():
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


def _context_lengths():
    for language in default_languages:
        print(f"language = {language}")
        x = load_x(language).to_numpy()
        print(
            f"average context length = {average([len(row[2].split(' ')) for row in x])}"
        )


if __name__ == "__main__":
    _results()
    # _examples()
    # _context_lengths()
