# Character RNN Text Generation

[![Open demo](https://img.shields.io/badge/Open%20demo-GitHub%20Pages-111827)](https://ymbawa26.github.io/cs443-project-4-rnn/)
[![Open in Colab](https://img.shields.io/badge/Open%20notebook-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ymbawa26/cs443-project-4-rnn/blob/main/rnn_text_generation.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-GRU-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

Recruiter-friendly readout of a CS 443 Bio-inspired Machine Learning project. The core idea: implement the machinery behind a character-level recurrent neural network instead of only calling a packaged RNN layer.

## What This Shows

- Implemented a custom Gated Recurrent Unit layer with update, reset, and candidate gates.
- Built a character-level preprocessing pipeline that converts raw reviews into overlapping fixed-length sequences.
- Added temporal cross-entropy loss that ignores padding tokens with a mask.
- Trained GRU RNNs that generate text one character at a time from a prompt.
- Completed and executed the assignment notebooks with validation output saved inline.

## Quick Results

| Check | Result |
| --- | --- |
| Dev corpus | Three Little Pigs character-level text |
| Vocabulary size | 39 characters |
| Training sequences | 14 sequences of length 103 |
| GRU_RNN1 final loss | `0.01070` after 220 epochs |
| Temporal loss test | `1.0988`, matching the expected notebook value |

Example generated continuation after the prompt `sticks. The wolf followed`:

```text
sticks. The wolf followed and se chinn the second the chinst little pig built a house of straw...
```

The output is not meant to be a production language model; it is a transparent implementation exercise showing how recurrent state, sampling, and training loss interact.

## What To Open First

- [Project demo page](https://ymbawa26.github.io/cs443-project-4-rnn/) gives the fastest overview.
- [RNN text generation notebook](https://github.com/ymbawa26/cs443-project-4-rnn/blob/main/rnn_text_generation.ipynb) shows the full model tests, training, and generation output.
- [Open the RNN notebook in Colab](https://colab.research.google.com/github/ymbawa26/cs443-project-4-rnn/blob/main/rnn_text_generation.ipynb) to run it online.

## Implementation

- `text_dataset_char.py`: loads reviews, cleans text, creates vocabularies, builds overlapping sequences, and converts characters to integer tokens.
- `rnn_layers.py`: custom GRU layer with recurrent state handling and padding masks.
- `rnn.py`: RNN model classes, temporal cross entropy, training/testing steps, and generation loop.
- `gru_layer.ipynb`: unit-style checks for GRU math and masking.
- `rnn_text_generation.ipynb`: end-to-end training and generation.

## Files

- `text_preprocessing.ipynb` covers character vocabulary, sequence creation, and integer encoding.
- `gru_layer.ipynb` tests the custom GRU implementation.
- `rnn_text_generation.ipynb` builds, trains, and samples from GRU RNNs.
- `text_dataset_char.py`, `rnn_layers.py`, and `rnn.py` contain the main Project 4 implementation.
- Supporting neural network utilities from earlier CS 443 projects are included.

## Data

`data/little_pigs.csv` is included for the small development corpus.

The full IMDb file is large, so it is not committed. Download it before running the full notebooks:

```bash
mkdir -p data
curl -L -o data/imdb_raw.csv https://cs.colby.edu/courses/S26/cs443/projects/p4rnn/data/imdb_raw.csv
```

Or run:

```bash
bash scripts/download_data.sh
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
jupyter notebook
```

## Validation

The notebooks were executed locally after completion. The Three Little Pigs GRU RNN overfit run reached a final training loss of about `0.01070`.
