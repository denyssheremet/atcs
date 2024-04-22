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
import pickle
from my_utils import load_from_pickle, save_to_pickle

parser = argparse.ArgumentParser(description='ATCS')
parser.add_argument("--encoder", type=int, default=0, help="Choose encoder (0=baseline, 1=lstm, 2=bi-lstm, 3=bi-lsm with max pooling)")
parser.add_argument("--dontsave", action='store_true', help="For debugging")
args = parser.parse_args()


# define the batch size
bs = 64

try:
    train_loader = load_from_pickle("pickle/train_loader.pickle")
    val_loader   = load_from_pickle("pickle/val_loader.pickle")
    test_loader  = load_from_pickle("pickle/test_loader.pickle")
    
    glove_vectors = load_from_pickle("pickle/glove_vectors.pickle")
    glove_vocab   = load_from_pickle("pickle/glove_vocab.pickle")
    
except: 
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
    
    # save to make it quicker next time
    save_to_pickle(train_loader, "pickle/train_loader.pickle")
    save_to_pickle(val_loader,   "pickle/val_loader.pickle")
    save_to_pickle(test_loader,  "pickle/test_loader.pickle")

# initialize the chosen encoder
if args.encoder == 0:
    print("Baseline")
    encoder = BaselineEncoder()
    mlp_in_dim = 1200
    checkpoint_path = "checkpoints/baseline_model.pickle"
elif args.encoder == 1:
    print("LSTM")
    encoder = LSTMEncoder()
    mlp_in_dim = 2048*4
    checkpoint_path = "checkpoints/lstm.pickle"
elif args.encoder == 2:
    print("BiLSTM")
    encoder = BiLSTMEncoder()
    mlp_in_dim = 2048*2*4    
    checkpoint_path = "checkpoints/bilstm.pickle"
elif args.encoder == 3:
    print("PooledBiLSTM")
    encoder = PooledBiLSTMEncoder()
    mlp_in_dim = 2048*2*4        
    checkpoint_path = "checkpoints/pooledbilstm.pickle"
else:
    raise Exception("incorrect choice of encoders. Look at the arguments.")
    
if args.dontsave: 
    print('***************************NO SAVING********************************')
    checkpoint_path = "checkpoints/tmp"

# create the model
model = FullModel(
    EmbeddingModule(glove_vectors, all_unique_toks),
    encoder,
    CombinationModule(),
    MLP(in_dim=mlp_in_dim),
).cuda()
print(model)

# create the loss module and optimizer
loss_module = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# create the tensorboard writer
writer = SummaryWriter()



# run the train loop
train_loop(
    model, 
    optimizer, 
    loss_module,
    train_loader, 
    val_loader,
    checkpoint_path,
    writer,
)


writer.close()

val_acc = evaluate(model, val_loader)
print(f'val acc: {val_acc}')

test_acc = evaluate(model, test_loader)
print(f'test acc: {test_acc}')