# SemEval-2020 Task 3: Graded Word Similarity in Context by Composing Pre-trained Embeddings

This repository holds the code and report written to fulfil the coursework requirements for the unit
[Dialogue and Narrative](https://www.bris.ac.uk/unit-programme-catalogue/UnitDetails.jsa?ayrCode=23%2F24&unitCode=COMSM0023)
at the University of Bristol.
The task I chose to investigate is
[SemEval-2020 Task 3: Graded Word Similarity in Context](https://aclanthology.org/2020.semeval-1.3/)
(Armendariz et al., SemEval 2020).

The data, which is reproduced in this repository, is [available here](https://competitions.codalab.org/competitions/20905).

## Instructions

> [!WARNING]
> This repository is designed to be used on macOS and has not been tested on other operating systems.

### Installation

Create a virtual environment and install Python dependencies:

```bash
conda create --name graded python=3.11
conda activate graded
conda install pip
pip install -r requirements.txt
```

Note that the default interpreter path in [settings.json](./.vscode/settings.json)
assumes you are using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
and the environment is named `graded`.

### Running the experiments

To run the experiments for subtask 1, either run the `subtask1` VS Code task in
[tasks.json](./.vscode/tasks.json) or execute the following command:

```bash
python -m src.subtask1 \
> --embedding static contextual pooled \
> --model-name bert-base-multilingual-cased \
> --language en fi hr sl \
> --window 0 1 2 3 \
> --operation concat none prod sum \
> --similarity cosine \
> --practice
```
