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
        model = MetaModel(
            params.embedding,
            params.model_name,
            params.window,
            params.operation,
            params.similarity,
        )
        score = model.score(x, y)
        time = perf_counter() - start

        directory = "results/predictions"
        makedirs(directory, exist_ok=True)
        DataFrame(
            {
                "predicted": model.predict(x),
                "actual": y,
            }
        ).to_csv(f"{directory}/{params.filename}", index=False)

    # pylint: disable=broad-exception-caught
    except Exception as exception:
        print(exception)

    if isnan(score):
        score = 0.0

    return score, time


def run_experiments():
    """Run the experiments."""

    args = parse_args()

    line()
    print(args)
    line()

    makedirs(args.directory, exist_ok=True)

    languages = args.language

    # There is no `practice kit' for Finnish.
    if args.practice and "fi" in languages:
        languages.remove("fi")

    results = []

    for language in languages:
        model_names = args.model_name
        if model_names == []:
            model_names = get_model_names(language)

        x = load_x(language, args.practice).to_numpy()
        y = load_y(language, args.practice).to_numpy()[:, 2]
        n = len(x)

        print(f"language = {language}, n = {n}")
        line()

        for params in product(
            args.embedding,
            model_names,
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

        DataFrame(results).to_csv(f"{args.directory}/{args.filename}", index=False)


if __name__ == "__main__":
    run_experiments()
