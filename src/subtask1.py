"""A script to run the subtask 1 experiments."""

import os
from math import isnan
from time import perf_counter

from pandas import DataFrame

from .data import languages, load_x, load_y
from .git import get_git_hash
from .models import ContextualBertModel, StaticBertModel
from .params import get_params

models: dict[str, type[ContextualBertModel | StaticBertModel]] = {
    # "static": StaticBertModel,
    "contextual": ContextualBertModel,
}


def main():
    """Run the subtask 1 experiments."""

    results_dir = f"results/{get_git_hash()}/subtask1"

    os.makedirs(results_dir, exist_ok=True)

    for name, model in models.items():
        for language in languages:
            x = load_x(language).to_numpy()
            y = load_y(language).to_numpy()[:, 2]
            n = len(x)

            print(f"{language} ({n} pairs)")
            print("-" * 80)

            results = []

            for (
                context_window_size,
                context_window_operation,
                similarity_measure,
                model_name,
            ) in get_params(language):
                print(f"{model_name} {context_window_operation} {context_window_size}")

                score = 0.0
                time = 0.0

                try:
                    start = perf_counter()
                    score = model(
                        context_window_size,
                        context_window_operation,
                        similarity_measure,
                        model_name,
                    ).score(x, y)
                    time = perf_counter() - start

                # pylint: disable=broad-exception-caught
                except Exception as exception:
                    print(exception)

                if isnan(score):
                    score = 0.0

                print(f"score = {score:.3f}")
                print(f"time = {time:.3f}")
                print(f"time/n = {(time / n):.3f}")
                print("-" * 80)

                results.append(
                    {
                        "model_name": model_name,
                        "context_window_operation": context_window_operation,
                        "context_window_size": context_window_size,
                        "similarity_measure": similarity_measure,
                        "score": score,
                        "time": time,
                        "time_per": time / n,
                    }
                )

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
