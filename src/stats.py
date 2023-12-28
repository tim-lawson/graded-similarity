"""Statistical tests."""

# pylint: disable=redefined-outer-name

from pandas import read_csv
from scipy.stats import ttest_rel


def mean_score(filename: str):
    """Mean score and variance."""

    dataframe = read_csv(filename)

    return dataframe["test_score"].mean(), dataframe["test_score"].var()


def test(filename1: str, filename2: str):
    """Dependent t-test for paired samples."""

    dataframe1 = read_csv(filename1)
    dataframe2 = read_csv(filename2)

    scores2 = dataframe2["test_score"].values
    scores1 = dataframe1["test_score"].values

    statistic, pvalue = ttest_rel(scores1, scores2)

    return statistic, pvalue
