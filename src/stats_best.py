"""
A script to test for statistically significant differences in the mean scores.
"""

# pylint: disable=redefined-outer-name


from pandas import DataFrame, concat

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
from .stats import mean_score, nemenyi_test, t_test

CV_SPLIT_TEST_SCORES = "results/cv/cv_split_test_scores_"


def save_t_test_results():
    """Save t-test results."""

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

    for params1, params2 in [
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
        mean1, variance1 = mean_score(CV_SPLIT_TEST_SCORES, params1)
        mean2, variance2 = mean_score(CV_SPLIT_TEST_SCORES, params2)

        statistic, pvalue = t_test(CV_SPLIT_TEST_SCORES, params1, params2)

        test_results = concat(
            [
                test_results,
                DataFrame.from_records(
                    [
                        {
                            "language": params1.language,
                            "embedding1": params1.embedding,
                            "model_name1": params1.model_name,
                            "window1": params1.window,
                            "operation1": params1.operation,
                            "mean1": mean1,
                            "variance1": variance1,
                            "embedding2": params2.embedding,
                            "model_name2": params2.model_name,
                            "window2": params2.window,
                            "operation2": params2.operation,
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

    test_results.to_csv("results/cv/cv_test_results_t.csv", index=False)


def save_nemenyi_test_results():
    """Save Nemenyi test results."""

    for language, paramss in [
        (
            "en",
            (
                en_static,
                en_contextual,
                en_pooled,
            ),
        ),
        (
            "fi",
            (
                fi_static,
                fi_contextual,
                fi_pooled,
            ),
        ),
        (
            "hr",
            (
                hr_static,
                hr_contextual,
                hr_pooled,
            ),
        ),
        (
            "sl",
            (
                sl_static,
                sl_contextual,
                sl_pooled,
            ),
        ),
    ]:
        test_results = nemenyi_test(CV_SPLIT_TEST_SCORES, paramss)

        test_results.to_csv(
            f"results/cv/cv_test_results_nemenyi_{language}.csv", index=False
        )


if __name__ == "__main__":
    save_t_test_results()
    save_nemenyi_test_results()
