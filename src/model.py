import math
import torch
import torch.nn as nn

from __transformer import *
from __rnn import *
from __lstm import *


class LMModel_transformer(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, device, dim=256, nhead=8, num_layers = 4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.embed_dim=dim
        self.n_head=nhead
        self.num_layers=num_layers
        self.device=device
        self.encoder = nn.Embedding(nvoc, dim, device=device)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        self.blocks=nn.Sequential()
        for _ in range(self.num_layers):
            self.blocks.add_module(f"block{_}",
                            TransformerDecoderBlock(embed_dim=self.embed_dim,
                                                       n_heads=self.n_head,
                                                       device=self.device))
        ########################################

        self.decoder = nn.Linear(dim, nvoc, device=device)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        #print(input.device)
        embeddings = self.drop(self.encoder(input))
        embeddings = positionencoding(embeddings)
        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        L = embeddings.size(1)
        src_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(input.device.type)
        src = embeddings * math.sqrt(self.embed_dim)
        for _, block in enumerate(self.blocks):
            src=block(src,src_mask)
        ########################################
        output = self.drop(src)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network
    3) Output linear layer
    """
    def __init__(self, nvoc, device, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.device,self.embed_dim,self.hidden_size,self.num_layers=device,dim,hidden_size,num_layers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim, device=self.device)
        ########################################
        self.rnn=nn.Sequential()
        for _ in range(self.num_layers):
            self.rnn.add_module(f"Block{_}",
                                RNNBlock(input_dim=self.embed_dim,
                                         hidden_dim=self.hidden_size,
                                         device=self.device))
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc, device=self.device)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        output=embeddings
        for _,block in enumerate(self.rnn):
            if _==0:
                output,hidden=block(output,hidden)
            else:
                output,hidden=block(output)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1))


class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    1) Embedding layer
    2) LSTM network
    3) Output linear layer
    """
    def __init__(self, nvoc, device, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.device,self.embed_dim,self.hidden_size,self.num_layers=device,dim,hidden_size,num_layers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim, device=device)
        ########################################
        self.lstm=nn.Sequential()
        for _ in range(num_layers):
            self.lstm.add_module(f"Block{_}",
                                 LSTMBlock(input_dim=dim,
                                           hidden_dim=hidden_size,
                                           device=device))
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc,device=device)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None, cell=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        output=embeddings
        for _,block in enumerate(self.lstm):
            if _==0:
                output=block(output,cell,hidden)
            else:
                output=block(output)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1))
