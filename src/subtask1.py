"""A script to run subtask 1 experiments."""

from itertools import product
from math import isnan
from time import perf_counter
from typing import Callable, Sequence

from numpy import ndarray
from pandas import DataFrame

from .args import parse_args
from .data import Language, default_languages, load_x, load_y
from .ensembles import get_contextual_ensemble, get_pooled_ensemble, get_static_ensemble
from .models import BaseModel, EnsembleModel, model_types
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
        score = model_types[params.model](
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


def run_experiments():
    """Run the experiments."""
    args = parse_args()

    line()
    print(args)
    line()

    results = []

    for language in args.language:
        x = load_x(language).to_numpy()
        y = load_y(language).to_numpy()[:, 2]
        n = len(x)

        print(f"language = {language}, n = {n}")
        line()

        for params in product(
            args.model,
            get_model_names(language),
            args.language,
            args.get_windows(),
            args.operation,
            args.similarity,
        ):
            params = Params(*params)
            print(params)

            score, time = run_experiment(x, y, params)

            results.append({**params.to_dict(), "score": score, "time": time})

            print(f"score = {score:.3f}")
            print(f"time = {n} x {(time / n):.6f} = {time:.3f} s")
            line()

        DataFrame(results).to_csv(f"results/{args.filename}", index=False)


def run_ensemble(
    filename: str,
    get_ensemble: Callable[[Language], Sequence[BaseModel]],
    weights: bool = False,
):
    """Run an ensemble experiment."""
    results = []

    for language in default_languages:
        x = load_x(language).to_numpy()
        y = load_y(language).to_numpy()[:, 2]
        n = len(x)

        print(f"language = {language}, n = {n}")

        start = perf_counter()
        score = EnsembleModel(get_ensemble(language), weights).score(x, y)
        time = perf_counter() - start

        results.append({"language": language, "score": score, "time": time})

        print(f"score = {score:.3f}")
        print(f"time = {n} x {(time / n):.6f} = {time:.3f} s")
        line()

    DataFrame(results).to_csv(f"results/ensembles/{filename}.csv", index=False)


def run_ensembles():
    """Run the ensemble experiments."""
    # run_ensemble("ensemble_static", get_static_ensemble)
    # run_ensemble("ensemble_static_weighted", get_static_ensemble, True)
    # run_ensemble("ensemble_contextual", get_contextual_ensemble)
    # run_ensemble("ensemble_contextual_weighted", get_contextual_ensemble, True)
    run_ensemble("ensemble_pooled", get_pooled_ensemble)
    run_ensemble("ensemble_pooled_weighted", get_pooled_ensemble, True)


if __name__ == "__main__":
    # run_experiments()
    run_ensembles()
