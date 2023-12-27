"""A script to run subtask 1 experiments."""

from itertools import product
from math import isnan
from os import makedirs
from time import perf_counter

from numpy import ndarray
from pandas import DataFrame

from .args import parse_args
from .data import load_x, load_y
from .models.meta import MetaModel
from .params import Params, get_model_names


def line():
    """Print a line."""
    print("-" * 80)


def run_experiment(
    x: ndarray,
    y: ndarray,
    params: Params,
):
    """Run an experiment."""
    score = 0.0
    time = 0.0

    try:
        start = perf_counter()
        score = MetaModel(
            params.model,
            params.model_name,
            params.window,
            params.operation,
            params.similarity,
        ).score(x, y)
        time = perf_counter() - start

    # pylint: disable=broad-exception-caught
    except Exception as exception:
        print(exception)

    if isnan(score):
        score = 0.0

    return score, time


def run_experiments(practice: bool = False):
    """Run the experiments."""

    args = parse_args()

    line()
    print(args)
    line()

    results_directory = "practice" if practice else "evaluation"
    makedirs(f"results/{results_directory}", exist_ok=True)

    languages = args.language

    # There is no `practice kit' for Finnish.
    if practice:
        languages.remove("fi")

    results = []

    for language in languages:
        x = load_x(language, practice).to_numpy()
        y = load_y(language, practice).to_numpy()[:, 2]
        n = len(x)

        print(f"language = {language}, n = {n}")
        line()

        for params in product(
            args.model,
            get_model_names(language),
            args.get_windows(),
            args.operation,
            args.similarity,
        ):
            params = Params(language, *params)
            print(params)

            score, time = run_experiment(x, y, params)

            results.append({**params.to_dict(), "score": score, "time": time})

            print(f"score = {score:.3f}")
            print(f"time = {n} x {(time / n):.6f} = {time:.3f} s")
            line()

        DataFrame(results).to_csv(
            f"results/{results_directory}/{args.filename}", index=False
        )


if __name__ == "__main__":
    run_experiments(practice=True)
