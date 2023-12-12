"""A script to run the subtask 1 experiments."""

import os
from math import isnan
from pprint import pprint
from time import perf_counter

from pandas import DataFrame

from .data import languages, load_x, load_y
from .git import get_git_hash
from .models import ContextualBertModel, StaticBertModel
from .params import get_params

models = {
    "static": lambda params: StaticBertModel(**params),
    "contextual": lambda params: ContextualBertModel(**params),
}


def main():
    """Run the subtask 1 experiments."""

    results_dir = f"results/{get_git_hash()}/subtask1"

    os.makedirs(results_dir, exist_ok=True)

    for name, model in models.items():
        for language in languages:
            print(f"{language}")
            print("-" * 80)

            x = load_x(language).to_numpy()
            y = load_y(language).to_numpy()[:, 2]

            results = []

            for params in get_params(language):
                pprint(params)

                score = 0.0
                time = 0.0

                try:
                    start = perf_counter()
                    score = model(dict(params)).score(x, y)
                    time = perf_counter() - start

                # pylint: disable=broad-exception-caught
                except Exception as exception:
                    print(exception)

                if isnan(score):
                    score = 0.0

                print(f"score: {score:.3f}")
                print(f"time: {time:.3f}")
                print("-" * 80)

                results.append({"score": score, "time": time, **params})

            results = sorted(
                results,
                key=lambda result: result["score"],
                reverse=True,
            )

            DataFrame(results).to_csv(
                f"{results_dir}/{name}_{language}.csv", index=False
            )


if __name__ == "__main__":
    main()
