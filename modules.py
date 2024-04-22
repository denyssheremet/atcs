from datasets import Dataset, load_dataset, get_dataset_split_names
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe, build_vocab_from_iterator
import pickle



class EmbeddingModule(nn.Module):
    def __init__(self, glove_vectors, all_unique_toks):
        super().__init__()
        
        # create the embedding layer and freeze the weights
        self.embedding_layer = nn.Embedding.from_pretrained(
            glove_vectors.get_vecs_by_tokens(["<unk>", *list(all_unique_toks)]),
            freeze=True
        )
    
    def forward(self, x):
        # unpack
        premises, hypotheses, labels = x
        # embed both the premise and hypothesis separately
        premises = self.embedding_layer(premises)
        hypotheses = self.embedding_layer(hypotheses)
        return (premises, hypotheses, labels)
    
class BaselineEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    # the baseline encoder takes the average across word embeddings in a sentence
    def forward(self, x):
        # unpack
        premises, hypotheses, labels = x
        premises = premises.mean(axis=1)
        hypotheses = hypotheses.mean(axis=1)
        return (premises, hypotheses, labels)
        

class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(300,2048,batch_first=True)
    
    # the baseline encoder takes the average across word embeddings in a sentence
    def forward(self, x):
        # unpack
        premises, hypotheses, labels = x
        premises = self.lstm.forward(premises)[1][0][0]
        hypotheses =  self.lstm.forward(hypotheses)[1][0][0]
        return (premises, hypotheses, labels)
        
class BiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(300,2048, batch_first=True, bidirectional=True)
    
    # the baseline encoder takes the average across word embeddings in a sentence
    def forward(self, x):
        # unpack
        premises, hypotheses, labels = x
        
        premises = self.lstm.forward(premises)[1][0]
        premises = torch.hstack([premises[0], premises[1]])
        
        hypotheses =  self.lstm.forward(hypotheses)[1][0]
        hypotheses = torch.hstack([hypotheses[0], hypotheses[1]])
        
        return (premises, hypotheses, labels)
        
        
class PooledBiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(300,2048, batch_first=True, bidirectional=True)
    
    # the baseline encoder takes the average across word embeddings in a sentence
    def forward(self, x):
        # unpack
        premises, hypotheses, labels = x
        
        premises = self.lstm.forward(premises)[0].max(dim=1)[0]
        hypotheses =  self.lstm.forward(hypotheses)[0].max(dim=1)[0]
    
        return (premises, hypotheses, labels)
    
        
class CombinationModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    # takes u and v (premise and hypothesis) 
    # and returns (u,v | u-v | u*v)
    def forward(self, x):
        # unpack
        u, v, labels = x
        out = torch.hstack([u,v,u-v,u*v])
        return out
    
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
class FullModel(nn.Module):
    def __init__(self, embedder, encoder, combinator, mlp):
        super().__init__()
        self.model = nn.ModuleList([
            embedder,
            encoder,
            combinator,
            mlp
        ])
    
    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x
    
    def p(self):
        print(self.model[1:])
    
    def save_model(self, filename):
        # torch.save(self.model[1:], filename)
        with open(filename, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self, filename):
        # layers = torch.load(filename)
        with open(filename, 'rb') as handle:
            self.model = pickle.load(handle)

