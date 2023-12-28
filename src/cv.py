"""
A script to run specific subtask 1 experiments and test for statistically significant
differences in the mean scores.
"""

# pylint: disable=redefined-outer-name

from os import makedirs

from pandas import DataFrame, concat
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GridSearchCV,
    ShuffleSplit,
)

from .data import Language, load_x, load_y
from .models.meta import MetaModel
from .models.utils import Embedding
from .params import Params, get_model_names
from .stats import mean_score, test

CV = int | BaseCrossValidator | BaseShuffleSplit | None


def _search_cv(
    language: Language,
    embedding: Embedding,
    model_name: str | None = None,
    window: int | None = None,
    operation: str | None = None,
    similarity: str | None = None,
    practice: bool = False,
    cv: CV = ShuffleSplit(n_splits=10, test_size=0.9, random_state=42),
):
    x = load_x(language, practice).to_numpy()
    y = load_y(language, practice).to_numpy()[:, 2]

    model_names = [model_name] if model_name is not None else get_model_names(language)

    max_window = 50 if embedding == "static" else 10

    windows = [window] if window is not None else list(range(max_window))

    operations = [operation] if operation is not None else ["sum", "prod", "concat"]

    similarities = [similarity] if similarity is not None else ["cosine"]

    search_cv = GridSearchCV(
        MetaModel(),
        param_grid={
            "model": [embedding],
            "model_name": model_names,
            "context_window_size": windows,
            "context_window_operation": operations,
            "similarity_measure": similarities,
        },
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


def _get_filename(
    language: Language,
    embedding: Embedding,
    model_name: str,
    window: int,
    operation: str,
    similarity: str = "cosine",
):
    return Params(
        language, embedding, model_name, window, operation, similarity
    ).filename


def _save_cv_result(
    language: Language,
    embedding: Embedding,
    model_name: str,
    window: int,
    operation: str,
):
    params, results, split_test_scores = _search_cv(
        language,
        embedding,
        model_name,
        window,
        operation,
    )

    results_dataframe = DataFrame.from_records(
        [{"language": language, **params, **results}]
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
            "language": language,
            **params,
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

    filename = _get_filename(
        language,
        embedding,
        model_name,
        window,
        operation,
    )

    makedirs("results/cv", exist_ok=True)

    results_dataframe.to_csv(f"results/cv/cv_results_{filename}", index=False)

    split_test_scores_dataframe.to_csv(
        f"results/cv/cv_split_test_scores_{filename}", index=False
    )


en_static = ("en", "static", "bert-large-uncased-whole-word-masking", 16, "sum")
en_contextual = ("en", "contextual", "bert-base-uncased", 1, "sum")
en_pooled = ("en", "pooled", "bert-base-uncased", 1, "sum")
fi_static = ("fi", "static", "EMBEDDIA/crosloengual-bert", 21, "sum")
fi_contextual = ("fi", "contextual", "TurkuNLP/bert-large-finnish-cased-v1", 1, "sum")
fi_pooled = ("fi", "pooled", "TurkuNLP/bert-large-finnish-cased-v1", 1, "sum")
hr_static = ("hr", "static", "classla/bcms-bertic", 31, "sum")
hr_contextual = ("hr", "contextual", "EMBEDDIA/crosloengual-bert", 3, "sum")
hr_pooled = ("hr", "pooled", "EMBEDDIA/crosloengual-bert", 3, "sum")
sl_static = ("sl", "static", "EMBEDDIA/crosloengual-bert", 11, "sum")
sl_contextual = ("sl", "contextual", "bert-base-multilingual-cased", 3, "sum")
sl_pooled = ("sl", "pooled", "EMBEDDIA/crosloengual-bert", 2, "sum")


def _save_cv_results():
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
        _save_cv_result(*params)


def _save_test_results():
    test_results = DataFrame(
        {
            "language": [],
            "embedding1": [],
            "model_name1": [],
            "window1": [],
            "operation1": [],
            "mean1": [],
            "variance1": [],
            "embedding2": [],
            "model_name2": [],
            "window2": [],
            "operation2": [],
            "mean2": [],
            "variance2": [],
            "statistic": [],
            "pvalue": [],
            "significant": [],
        }
    )

    for model1, model2 in [
        (en_contextual, en_static),
        (en_pooled, en_static),
        (en_pooled, en_contextual),
        (fi_contextual, fi_static),
        (fi_pooled, fi_static),
        (fi_pooled, fi_contextual),
        (hr_contextual, hr_static),
        (hr_pooled, hr_static),
        (hr_pooled, hr_contextual),
        (sl_contextual, sl_static),
        (sl_pooled, sl_static),
        (sl_pooled, sl_contextual),
    ]:
        filename1 = f"results/cv/cv_split_test_scores_{_get_filename(*model1)}"
        filename2 = f"results/cv/cv_split_test_scores_{_get_filename(*model2)}"

        mean1, variance1 = mean_score(filename1)
        mean2, variance2 = mean_score(filename2)

        statistic, pvalue = test(filename1, filename2)

        test_results = concat(
            [
                test_results,
                DataFrame.from_records(
                    [
                        {
                            "language": model1[0],
                            "embedding1": model1[1],
                            "model_name1": model1[2],
                            "window1": model1[3],
                            "operation1": model1[4],
                            "mean1": mean1,
                            "variance1": variance1,
                            "embedding2": model2[1],
                            "model_name2": model2[2],
                            "window2": model2[3],
                            "operation2": model2[4],
                            "mean2": mean2,
                            "variance2": variance2,
                            "statistic": statistic,
                            "pvalue": pvalue,
                            "significant": pvalue < 0.05,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    test_results["significant"] = test_results["significant"].astype(bool)

    test_results.to_csv("results/cv/cv_test_results.csv", index=False)


if __name__ == "__main__":
    _save_cv_results()
    _save_test_results()
