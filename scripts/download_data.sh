#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
curl -L -o data/imdb_raw.csv https://cs.colby.edu/courses/S26/cs443/projects/p4rnn/data/imdb_raw.csv
curl -L -o data/little_pigs.csv https://cs.colby.edu/courses/S26/cs443/projects/p4rnn/data/little_pigs.csv
