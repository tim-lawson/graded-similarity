"""Utilities to load and pre-process data."""

from typing import Literal

from pandas import read_csv  # type: ignore

Language = Literal[
    "en",
    "fi",
    "hr",
    "sl",
]

default_languages: list[Language] = [
    "en",
    "fi",
    "hr",
    "sl",
]


def load_x(language: Language):
    """Load the evaluation data."""

    x = read_csv(f"./data/evaluation_kit_final/data/data_{language}.tsv", sep="\t")
    x["context1"] = x["context1"].apply(remove_strong_tags)
    x["context2"] = x["context2"].apply(remove_strong_tags)
    return x


def load_y(language: Language):
    """Load the gold-standard values."""

    return read_csv(f"./data/gold/gold_{language}.tsv", sep="\t")


def remove_strong_tags(value: str) -> str:
    """Remove <strong> tags."""

    return value.replace("<strong>", "").replace("</strong>", "")
