# atcs

### Installation
Run `pip install -r requirements.txt`
requirements:
 - torch
 - datasets
 - torchtext

### Training
To run the training, run `python train.py` in the terminal. 
Arguments: 
 - `--encoder` (int): 0=baseline, 1=lstm, 2=bi-lstm, 3=bi-lsm with max pooling

### Evaluation
For evaluation, run `python eval.py` in the terminal.
Arguments:
 - `--dataset` (string): options: "snli" or "senteval".

Example: 
To reproduce the results on the SNLI dataset, run `python eval.py --dataset snli`.

### Code structure
The code for running the training is in `train.py`. The code for evaluation is in `eval.py`.
Helper functions for preprocessing are in `preprocessing.py`.
Functions for the training and evaluation loop are in `train_functions.py`.
And the components of the pipeline (Embedding module, encoders, combination module and MLP) are in `modules.py`.
