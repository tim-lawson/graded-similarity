"""A script to run hyperparameter search/'cross-validation'."""

# pylint: disable=redefined-outer-name

from itertools import product

from pandas import DataFrame, concat
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    RandomizedSearchCV,
    ShuffleSplit,
)

from .data import Language, default_languages, load_x, load_y
from .models.meta import MetaModel
from .models.utils import Model, models
from .params import get_model_names


def search(
    language: Language,
    model: Model = "static",
    cv: int
    | BaseCrossValidator
    | BaseShuffleSplit
    | None = ShuffleSplit(n_splits=10, test_size=0.9, random_state=42),
    n_iter: int = 100,
    n_jobs: int = 2,
):
    """Hyperparameter search for a language and model."""

    x = load_x(language, True).to_numpy()
    y = load_y(language, True).to_numpy()[:, 2]

    search_cv = RandomizedSearchCV(
        MetaModel(),
        param_distributions={
            "model": [model],
            "model_name": get_model_names(language),
            "context_window_size": list(range(50 if model == "static" else 10)),
            "context_window_operation": ["sum", "prod", "concat"],
            "similarity_measure": ["cosine"],
        },
        n_iter=n_iter,
        cv=cv,
        verbose=4,
        n_jobs=n_jobs,
    ).fit(x, y)

    index = search_cv.best_index_

    results = {
        "best_score": search_cv.best_score_,
        "mean_fit_time": search_cv.cv_results_["mean_fit_time"][index],
        "std_fit_time": search_cv.cv_results_["std_fit_time"][index],
        "mean_score_time": search_cv.cv_results_["mean_score_time"][index],
        "std_score_time": search_cv.cv_results_["std_score_time"][index],
    }

    split_test_scores: list[float] = []

    for index in range(search_cv.n_splits_):
        split_test_scores.append(
            search_cv.cv_results_[f"split{index}_test_score"][index]
        )

    return search_cv.best_params_, results, split_test_scores


if __name__ == "__main__":
    results_dataframe = DataFrame(
        columns=[
            "language",
            "model",
            "model_name",
            "context_window_size",
            "context_window_operation",
            "similarity_measure",
            "best_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        ]
    )

    split_test_scores_dataframe = DataFrame(
        columns=[
            "language",
            "model",
            "model_name",
            "context_window_size",
            "context_window_operation",
            "similarity_measure",
            "split",
            "test_score",
        ]
    )

    for language, model in product(default_languages, models):
        params, results, split_test_scores = search(language, model)

        print(f"{language} {model} {results['best_score']:.3f}")

        results_dataframe = concat(
            [
                results_dataframe,
                DataFrame.from_records(
                    [
                        {
                            **params,
                            **results,
                            "language": language,
                            "model": model,
                        }
                    ]
                ),
            ],
        )

        split_test_scores_dataframe = concat(
            [
                split_test_scores_dataframe,
                DataFrame.from_records(
                    {
                        **params,
                        "language": language,
                        "model": model,
                        "split": list(range(len(split_test_scores))),
                        "test_score": split_test_scores,
                    }
                ),
            ],
            ignore_index=True,
        )

    results_dataframe.to_csv("src/cv_results.csv", index=False)

    split_test_scores_dataframe.to_csv("src/cv_split_test_scores.csv", index=False)
