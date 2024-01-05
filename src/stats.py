"""Statistical tests."""

# pylint: disable=redefined-outer-name

from numpy import array
from pandas import read_csv
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare, ttest_rel

from .params import Params


def mean_score(prefix: str, params: Params):
    """Mean score and variance."""

    dataframe = read_csv(prefix + params.filename)

    return dataframe["test_score"].mean(), dataframe["test_score"].var()


def read(prefix: str, params: Params):
    """Read dataframe for a given experiment."""

    return read_csv(prefix + params.filename)["test_score"].values


def t_test(prefix: str, params1: Params, params2: Params):
    """Dependent t-test for two paired samples."""

    statistic, pvalue = ttest_rel(read(prefix, params1), read(prefix, params2))

    return statistic, pvalue


def nemenyi_test(prefix: str, paramss: tuple[Params, ...]):
    """Nemenyi test for N paired samples."""

    if len(set(params.language for params in paramss)) > 1:
        raise ValueError("Samples must be from the same language")

    scores = [read(prefix, params) for params in paramss]

    _statistic, pvalue = friedmanchisquare(*scores)

    if pvalue >= 0.05:
        raise ValueError("Difference between samples is not significant")

    test_results = posthoc_nemenyi_friedman(array(scores).T)

    # TODO: don't assume that the comparative parameter is `embedding`
    columns = [params.embedding for params in paramss]

    test_results.columns = columns
    test_results.index = test_results.columns

    # Convert to one row per comparison
    test_results = test_results.stack().reset_index()

    test_results.columns = ["embedding1", "embedding2", "pvalue"]

    # Remove self-comparisons
    test_results = test_results[
        test_results["embedding1"] != test_results["embedding2"]
    ]

    # Remove duplicates
    test_results = test_results[
        test_results.apply(lambda row: row["embedding1"] < row["embedding2"], axis=1)
    ]

    test_results["significant"] = test_results["pvalue"] < 0.05
    test_results["significant"] = test_results["significant"].astype(bool)

    def find_params(row, column: str):
        return [params for params in paramss if params.embedding == row[column]][0]

    def tstatistic(row):
        statistic, _pvalue = t_test(
            prefix, find_params(row, "embedding1"), find_params(row, "embedding2")
        )

        return statistic

    test_results["tstatistic"] = test_results.apply(tstatistic, axis=1)

    def score(column: str):
        return lambda row: mean_score(prefix, find_params(row, column))[0]

    test_results["score1"] = test_results.apply(score("embedding1"), axis=1)
    test_results["score2"] = test_results.apply(score("embedding2"), axis=1)

    test_results["embedding1"] = test_results["embedding1"].str.capitalize()
    test_results["embedding2"] = test_results["embedding2"].str.capitalize()

    test_results = test_results[
        [
            "embedding1",
            "score1",
            "embedding2",
            "score2",
            "tstatistic",
            "pvalue",
            "significant",
        ]
    ]

    return test_results
