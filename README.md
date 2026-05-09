# CS 443 Project 4: Recurrent Neural Networks

Character-level recurrent neural network work for CS 443 Bio-inspired Machine Learning.

This project implements:

- character-level text preprocessing for IMDb-style reviews
- a custom Gated Recurrent Unit layer in TensorFlow
- temporal cross-entropy loss with padding masks
- GRU-based RNN text generation
- completed Jupyter notebooks with validation outputs

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

## Validation

The notebooks were executed locally after completion. The Three Little Pigs GRU RNN overfit run reached a final training loss of about `0.01070`.
