"""Utilities to load and pre-process data."""

from typing import Literal

from pandas import read_csv

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


def load_x(language: Language, practice: bool = False):
    """Load the evaluation data."""

    prefix = "practice" if practice else "evaluation"
    x = read_csv(
        f"./data/{prefix}_kit_final/data/data_{language}.tsv",
        sep="\t",
    )
    x["context1"] = x["context1"].apply(remove_strong_tags)
    x["context2"] = x["context2"].apply(remove_strong_tags)
    return x


def load_y(language: Language, practice: bool = False):
    """Load the gold-standard values."""

    prefix = "practice_kit_final/" if practice else ""
    return read_csv(f"./data/{prefix}gold/gold_{language}.tsv", sep="\t")


def load(language: Language):
    """Load the data."""

    x = load_x(language)
    y = load_y(language)
    return x, y


def remove_strong_tags(value: str) -> str:
    """Remove <strong> tags."""

    return value.replace("<strong>", "").replace("</strong>", "")
