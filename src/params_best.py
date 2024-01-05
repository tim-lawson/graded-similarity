from .params import Params

en_static = Params(
    "en",
    "static",
    "bert-large-uncased-whole-word-masking",
    16,
    "sum",
)

en_contextual = Params(
    "en",
    "contextual",
    "bert-base-uncased",
    1,
    "sum",
)

en_pooled = Params(
    "en",
    "pooled",
    "bert-base-uncased",
    1,
    "sum",
)

fi_static = Params(
    "fi",
    "static",
    "EMBEDDIA/crosloengual-bert",
    21,
    "sum",
)

fi_contextual = Params(
    "fi",
    "contextual",
    "TurkuNLP/bert-large-finnish-cased-v1",
    1,
    "sum",
)

fi_pooled = Params(
    "fi",
    "pooled",
    "TurkuNLP/bert-large-finnish-cased-v1",
    1,
    "sum",
)

hr_static = Params(
    "hr",
    "static",
    "classla/bcms-bertic",
    31,
    "sum",
)

hr_contextual = Params(
    "hr",
    "contextual",
    "EMBEDDIA/crosloengual-bert",
    3,
    "sum",
)

hr_pooled = Params(
    "hr",
    "pooled",
    "EMBEDDIA/crosloengual-bert",
    3,
    "sum",
)

sl_static = Params(
    "sl",
    "static",
    "EMBEDDIA/crosloengual-bert",
    11,
    "sum",
)

sl_contextual = Params(
    "sl",
    "contextual",
    "bert-base-multilingual-cased",
    3,
    "sum",
)

sl_pooled = Params(
    "sl",
    "pooled",
    "EMBEDDIA/crosloengual-bert",
    2,
    "sum",
)
