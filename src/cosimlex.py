"""A script to convert the CoSimLex dataset to the evaluation format."""

import os

from pandas import DataFrame, read_csv

from .data import default_languages


def convert():
    """Convert the CoSimLex dataset to the evaluation format."""
    for language in default_languages:
        cosimlex = read_csv(
            f"./data/cosimlex_dataset/cosimlex_{language}.csv", sep="\t"
        )
        data = DataFrame(
            {
                "sim_context1": cosimlex["sim1"],
                "sim_context2": cosimlex["sim2"],
                "change": cosimlex["sim2"] - cosimlex["sim1"],
            }
        )
        os.makedirs("./data/gold", exist_ok=True)
        data.to_csv(f"./data/gold/gold_{language}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    convert()
