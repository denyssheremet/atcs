from datasets import Dataset, load_dataset, get_dataset_split_names
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe, build_vocab_from_iterator
from preprocessing import lower_and_tokenize, get_unique_tokens, setup_glove, tokens_to_indexes_and_filter_labels, collate_fn
from modules import EmbeddingModule, BaselineEncoder, CombinationModule, MLP, FullModel, LSTMEncoder, BiLSTMEncoder, PooledBiLSTMEncoder
from train_functions import train_epoch, evaluate, update_lr, train_loop
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser(description='ATCS')
parser.add_argument("--encoder", type=int, default=0, help="Choose encoder (0=baseline, 1=lstm, 2=bi-lstm, 3=bi-lsm with max pooling)")
parser.add_argument("--dataset", type=str, default='snli', help="Choose dataset (snli or senteval)")

args = parser.parse_args()


# define the batch size
bs = 64

# load the dataset
ds_train, ds_val, ds_test = load_dataset("stanfordnlp/snli", split=['train', 'validation', 'test'])

# convert to lowercase and tokenize
ds_train_tok, ds_val_tok, ds_test_tok = lower_and_tokenize(ds_train, ds_val, ds_test)

# get a list of all the unique tokens in the dataset
all_unique_toks = get_unique_tokens(ds_train_tok, ds_val_tok, ds_test_tok)

# set up the glove vectors and vocabulary
glove_vectors, glove_vocab = setup_glove(all_unique_toks)

# convert the tokens to indices, and remove items with label -1
ds_train_prep, ds_val_prep, ds_test_prep = tokens_to_indexes_and_filter_labels(ds_train_tok, ds_val_tok, ds_test_tok, glove_vocab)

# create the dataloaders
train_loader = DataLoader(ds_train_prep, collate_fn=collate_fn, batch_size=bs, shuffle=True)
val_loader   = DataLoader(ds_val_prep, collate_fn=collate_fn, batch_size=bs, shuffle=True)
test_loader  = DataLoader(ds_test_prep, collate_fn=collate_fn, batch_size=bs, shuffle=True)

# initialize the chosen encoder
if args.encoder == 0:
    print("Baseline")
    encoder = BaselineEncoder()
    mlp_in_dim = 1200
    checkpoint_path = "baseline_model.pickle"
elif args.encoder == 1:
    print("LSTM")
    encoder = LSTMEncoder()
    mlp_in_dim = 1200
    checkpoint_path = "lstm.pickle"
elif args.encoder == 2:
    print("BiLSTM")
    encoder = BiLSTMEncoder()
    mlp_in_dim = 2400    
    checkpoint_path = "bilstm.pickle"
elif args.encoder == 3:
    print("PooledBiLSTM")
    encoder = PooledBiLSTMEncoder()
    mlp_in_dim = 2400    
    checkpoint_path = "pooledbilstm.pickle"
else:
    raise Exception("incorrect choice of encoders. Look at the arguments.")
    

# create the model
model = FullModel(
    EmbeddingModule(glove_vectors, all_unique_toks),
    encoder,
    CombinationModule(),
    MLP(in_dim=mlp_in_dim),
).cuda()
model.load_model(checkpoint_path)
print(model)
print()

val_acc = evaluate(model, val_loader)
print(f'val acc: {val_acc}')

test_acc = evaluate(model, test_loader)
print(f'test acc: {test_acc}')

