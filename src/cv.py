"""
A script to run specific subtask 1 experiments.
"""

# pylint: disable=redefined-outer-name

from os import makedirs

from pandas import DataFrame
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GridSearchCV,
    ShuffleSplit,
)

from .data import load_x, load_y
from .models.meta import MetaModel
from .params import Params
from .params_best import (
    en_contextual,
    en_pooled,
    en_static,
    fi_contextual,
    fi_pooled,
    fi_static,
    hr_contextual,
    hr_pooled,
    hr_static,
    sl_contextual,
    sl_pooled,
    sl_static,
)

CV = int | BaseCrossValidator | BaseShuffleSplit | None


def param_grid_params(params: Params):
    """Parameter grid from Params."""

    return {
        "model": [params.embedding],
        "model_name": [params.model_name],
        "context_window_size": [params.window],
        "context_window_operation": [params.operation],
        "similarity_measure": [params.similarity],
    }


def search_cv(
    params: Params,
    practice: bool = False,
    cv: CV = ShuffleSplit(n_splits=10, test_size=0.9, random_state=42),
):
    """Hyperparameter search over cross-validation folds."""

    x = load_x(params.language, practice).to_numpy()
    y = load_y(params.language, practice).to_numpy()[:, 2]

    search_cv = GridSearchCV(
        MetaModel(),
        param_grid=param_grid_params(params),
        cv=cv,
        verbose=4,
    ).fit(x, y)

    best_index = search_cv.best_index_

    results = {
        "best_score": search_cv.best_score_,
        "mean_fit_time": search_cv.cv_results_["mean_fit_time"][best_index],
        "std_fit_time": search_cv.cv_results_["std_fit_time"][best_index],
        "mean_score_time": search_cv.cv_results_["mean_score_time"][best_index],
        "std_score_time": search_cv.cv_results_["std_score_time"][best_index],
    }

    split_test_scores: list[float] = []

    for split_index in range(search_cv.n_splits_):
        split_test_scores.append(
            search_cv.cv_results_[f"split{split_index}_test_score"][best_index]
        )

    return search_cv.best_params_, results, split_test_scores


def save_cv_result(params: Params):
    """Save cross-validation results."""

    best_params, results, split_test_scores = search_cv(params)

    results_dataframe = DataFrame.from_records(
        [{"language": params.language, **best_params, **results}]
    )

    results_dataframe.columns = [
        "language",
        "embedding",
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

    split_test_scores_dataframe = DataFrame.from_records(
        {
            "language": params.language,
            **best_params,
            "split": list(range(len(split_test_scores))),
            "test_score": split_test_scores,
        }
    )

    split_test_scores_dataframe.columns = [
        "language",
        "embedding",
        "model_name",
        "context_window_size",
        "context_window_operation",
        "similarity_measure",
        "split",
        "test_score",
    ]

    makedirs("results/cv", exist_ok=True)

    filename = params.filename

    results_dataframe.to_csv(f"results/cv/cv_results_{filename}", index=False)

    split_test_scores_dataframe.to_csv(
        f"results/cv/cv_split_test_scores_{filename}", index=False
    )


def save_cv_results():
    """Save cross-validation results."""

    for params in [
        en_static,
        en_contextual,
        en_pooled,
        fi_static,
        fi_contextual,
        fi_pooled,
        hr_static,
        hr_contextual,
        hr_pooled,
        sl_static,
        sl_contextual,
        sl_pooled,
    ]:
        save_cv_result(params)


if __name__ == "__main__":
    save_cv_results()
