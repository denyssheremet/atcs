from datasets import Dataset, load_dataset, get_dataset_split_names
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe, build_vocab_from_iterator
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pickle


def preprocess_item(item):
    item['premise'] = word_tokenize(item['premise'].lower())
    item['hypothesis'] = word_tokenize(item['hypothesis'].lower())
    return item


def preprocess_item(item):
    item['premise'] = word_tokenize(item['premise'].lower())
    item['hypothesis'] = word_tokenize(item['hypothesis'].lower())
    return item

# get a list of all the unique tokens
def get_all_unique_toks(all_ds_tok):
    all_unique_toks = set()
    for ds in all_ds_tok:
        for item in tqdm.tqdm(ds):
            for key in ['premise', 'hypothesis']:
                for tok in item[key]:
                    if not tok in all_unique_toks:
                        all_unique_toks.add(tok)
    return all_unique_toks


def token_to_index(item, glove_vocab):
    item['premise'] = glove_vocab.lookup_indices(item['premise'])    
    item['hypothesis'] = glove_vocab.lookup_indices(item['hypothesis'])
    return item


def lower_and_tokenize(ds_train, ds_val, ds_test):
    # lower and tokenize
    try: 
        with open('all_ds_tok.pickle', 'rb') as handle:
            all_ds_tok = pickle.load(handle)

        ds_train_tok = all_ds_tok[0]
        ds_val_tok   = all_ds_tok[1]
        ds_test_tok  = all_ds_tok[2]
    except: 
        ds_train_tok = ds_train.map(preprocess_item)
        ds_val_tok = ds_val.map(preprocess_item)
        ds_test_tok = ds_test.map(preprocess_item)
        all_ds_tok = [ds_train_tok, ds_val_tok, ds_test_tok]

        with open('all_ds_tok.pickle', 'wb') as handle:
            pickle.dump(all_ds_tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ds_train_tok, ds_val_tok, ds_test_tok

def get_unique_tokens(ds_train_tok, ds_val_tok, ds_test_tok):
    # get list of unique tokens
    try: 
        with open('all_unique_toks.pickle', 'rb') as handle:
            all_unique_toks = pickle.load(handle)
    except: 
        all_unique_toks = get_all_unique_toks([ds_train_tok, ds_val_tok, ds_test_tok])
        with open('all_unique_toks.pickle', 'wb') as handle:
            pickle.dump(all_unique_toks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return all_unique_toks

def setup_glove(all_unique_toks):
    try: 
        with open('glove_vectors.pickle', 'rb') as handle:
            glove_vectors = pickle.load(handle)
        print('loading glove vectors...')
    except: 
        glove_vectors = GloVe(name='840B')
        with open('glove_vectors.pickle', 'wb') as handle:
            pickle.dump(glove_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # glove vocab     
    try: 
        with open('glove_vocab.pickle', 'rb') as handle:
            glove_vocab = pickle.load(handle)
        print('loading glove vocab...')
        
    except: 
        glove_vocab = build_vocab_from_iterator(
            [iter(all_unique_toks)],
            specials=['<unk>'],
            special_first=True,
        )
        with open('glove_vocab.pickle', 'wb') as handle:
            pickle.dump(glove_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return glove_vectors, glove_vocab


def tokens_to_indexes_and_filter_labels(ds_train_tok, ds_val_tok, ds_test_tok, glove_vocab):
    # convert tokens to indices, and remove items with label -1
    try: 
        with open('all_ds_prep.pickle', 'rb') as handle:
            all_ds_prep = pickle.load(handle)
        ds_train_prep = all_ds_prep[0]
        ds_val_prep = all_ds_prep[1]
        ds_test_prep = all_ds_prep[2]
    except: 
        # map to index and remove items with label -1
        ds_train_prep = ds_train_tok.map(lambda x: token_to_index(x, glove_vocab)).filter(lambda x: x['label'] >= 0)
        ds_val_prep   = ds_val_tok.map(lambda x: token_to_index(x, glove_vocab)).filter(lambda x: x['label'] >= 0)
        ds_test_prep  = ds_test_tok.map(lambda x: token_to_index(x, glove_vocab)).filter(lambda x: x['label'] >= 0)
        all_ds_prep = [ds_train_prep, ds_val_prep, ds_test_prep]

        with open('all_ds_prep.pickle', 'wb') as handle:
            pickle.dump(all_ds_prep, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return ds_train_prep, ds_val_prep, ds_test_prep

def collate_fn(data):
    # first find the longest sentences in the batch
    max_p, max_h = -1, -1
    for d in data:
        max_p = max(max_p, len(d['premise']))
        max_h = max(max_h, len(d['hypothesis']))
    
    # pad all sentences to same length
    bs = len(data)
    batch_p = torch.zeros((bs,max_p), dtype=torch.int32)
    batch_h = torch.zeros((bs,max_h), dtype=torch.int32)
    batch_l = torch.zeros((bs), dtype=torch.int64)
    
    for i, d in enumerate(data):
        batch_p[i, 0:len(d['premise'])] = torch.tensor(d['premise'])
        batch_h[i, 0:len(d['hypothesis'])] = torch.tensor(d['hypothesis'])
        batch_l[i] = d['label']
        
    return batch_p, batch_h, batch_l