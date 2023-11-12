"""A script to run the subtask 1 experiments."""

import os
from pprint import pprint
from time import perf_counter
from typing import Any, Callable, Sequence

from pandas import DataFrame

from .data import languages, load_x, load_y
from .git import get_git_hash
from .graded_similarity import (
    BaseGradedSimilarity,
    ContextualGradedSimilarity,
    ContextualGradedSimilarityParams,
    StaticGradedSimilarity,
    StaticGradedSimilarityParams,
)
from .params import contextual_params, static_params


def run(
    graded_similarity: Callable[[dict[str, Any]], BaseGradedSimilarity],
    params_list: Sequence[
        StaticGradedSimilarityParams | ContextualGradedSimilarityParams
    ],
    results_dir: str,
    results_name: str,
):
    """Run a subtask 1 experiment."""

    for language in languages:
        print(f"{language}")
        print("-" * 80)

        x = load_x(language).to_numpy()
        y = load_y(language).to_numpy()[:, 2]

        results = []
        for params in params_list:
            pprint(params)

            start = perf_counter()
            score = graded_similarity(dict(params)).score(x, y)
            time = perf_counter() - start

            print(f"score: {score:.3f}")
            print(f"time: {time:.3f}")
            print("-" * 80)

            results.append({"score": score, "time": time, **params})

        # Sort the results in descending order of score.
        results = sorted(
            results,
            key=lambda result: result["score"],
            reverse=True,
        )

        DataFrame(results).to_csv(
            f"{results_dir}/{results_name}_{language}.csv", index=False
        )


if __name__ == "__main__":
    # Create a results directory for the current commit hash.
    results_hash_dir = f"results/{get_git_hash()}/subtask1"

    os.makedirs(results_hash_dir, exist_ok=True)

    run(
        lambda params: StaticGradedSimilarity(**params),
        static_params,
        results_hash_dir,
        "static",
    )

    run(
        lambda params: ContextualGradedSimilarity(**params),
        contextual_params,
        results_hash_dir,
        "contextual",
    )
