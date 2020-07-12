# SAttParser

Submission to EMNLP 2020

Constituency parsing with Span attention

## Prerequisites
* python 3.6
* pytorch 1.1

Install python dependencies by running:

`
pip install -r requirements.txt
`

Models are trained on a single Nvidia Tesla V100 GPU with 16G RAM.


To train a model on a small dataset, run:

`
./run.sh
`


The hyper-parameters for the best performing models are reported in the "./run.sh" file.

