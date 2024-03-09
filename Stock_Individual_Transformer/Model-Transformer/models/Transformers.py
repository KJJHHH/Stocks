import torch
import torchvision
import torch.nn as nn

import math 
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    Why odd numer of d_model not work
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderDecoder(nn.Module):

    def __init__(
        self, 
        d_model: int = 6,  nhead: int = 6,  dropout: float = 0.5,
        d_hid: int = int(256/2), num_class: int = 1, nlayers_e: int = int(64*2), 
        nlayers_d: int = int(16*1), windows: int = 10, ntoken: int = 100):        
        super().__init__()
        
        self.model_type = f'TransEnDecoder-Window{windows}EL{nlayers_e}DL{nlayers_d}Hid{d_hid}'
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers_e)
        decoder_layers = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers_d)
        
        # Convolution 
        """
        # Use in version 1, no 2
        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()        
        self.conv_linear1 = nn.Linear(6*100, d_model)
        self.conv_linear2 = nn.Linear(16*48, d_model)
        self.conv_linear3 = nn.Linear(32*22, d_model)
        self.conv_linear4 = nn.Linear(64*9, d_model)
        self.conv_linear5 = nn.Linear(64*3, d_model)
        self.conv_linear = nn.Linear(64, d_model)
        """
        
        # Fin
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model*windows, num_class)
        self.linear2 = nn.Linear(d_model*windows, num_class)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, train: bool, 
                memory: Tensor = None, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        if src_mask is None:
            Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        # See the input as encoding 
        # src = self.embedding(src) * math.sqrt(self.d_model)
        
        # Check if train is False and memory is None
        assert train or memory is not None, "Test mode but no memory"
        # Positional encode
        """
        Input of pos_encoding:
        src: (totallen, seqlen, d_model)
        tgt: (batch, seqlen, d_model)
        """        
        src = self.pos_encoder(src) 
        if tgt.size(1) > 1:
            tgt = self.pos_encoder(tgt)
        
        # Lengths
        N = src.size(0)
        L_src = src.size(1)
        L_tgt = tgt.size(1)
        D = src.size(2)
        
        # Encoder
        if train is True: 
            """
            Output of encoder: (1, seqlen, d_model)
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(L_src).to(device)
            memory = self.transformer_encoder(src, src_mask) # Memory  
            memory = memory.repeat(tgt.size(0), 1, 1)
        
        # Decoder
        # =====
        """
        # If padding mask:
        tgt_padding_mask = self.padding_mask(tgt).to(device)
        output = self.transformer_decoder(
            tgt=tgt, memory=memory , tgt_key_padding_mask = tgt_padding_mask) # 
        """
        output = self.transformer_decoder(tgt=tgt, memory=memory) 
        output = tgt + output
        output = output.reshape(output.size(0), -1)
        output = self.linear1(output)
        
        tgt = self.linear2(tgt.reshape(output.size(0), -1))
        output = tgt + output
        return memory, output
        
    def transform_patch_len_to_1(self, tgt):
        """
        Use in version 1, no 2
        """
        tgt = tgt.permute(0, 2, 1)
        tgt1 = tgt.view(tgt.size(0), -1)
        tgt = self.conv1(tgt)        
        tgt = self.bn1(tgt)
        tgt = self.maxpool(tgt)
        tgt = self.relu(tgt)
        tgt2 = tgt.view(tgt.size(0), -1)        
        tgt = self.conv2(tgt)
        tgt = self.bn2(tgt)
        tgt = self.maxpool(tgt)
        tgt = self.relu(tgt)
        tgt3 = tgt.view(tgt.size(0), -1)
        tgt = self.conv3(tgt)
        tgt = self.bn3(tgt)
        tgt = self.maxpool(tgt)
        tgt = self.relu(tgt)
        tgt4 = tgt.view(tgt.size(0), -1)
        tgt = self.maxpool(tgt)
        tgt5 = tgt.view(tgt.size(0), -1)
        tgt = self.maxpool2(tgt)
        tgt = tgt.view(tgt.size(0), -1)
        tgt = self.conv_linear(tgt)
        tgt1 = self.conv_linear1(tgt1)
        tgt2 = self.conv_linear2(tgt2)
        tgt3 = self.conv_linear3(tgt3)
        tgt4 = self.conv_linear4(tgt4)
        tgt5 = self.conv_linear5(tgt5)
        tgt = tgt + tgt1 + tgt2 + tgt3 + tgt4 + tgt5        
        
        tgt = tgt.unsqueeze(1)
        
        return tgt
        
    def padding_mask(self, x):
        N = x.size(0) # batch size
        L = x.size(1) # sequence length        
        patch_mask = torch.randint(0, int(L*0.5), (x.size(0),), dtype=torch.int)
        padding_mask = torch.zeros(N, L, dtype=torch.float32)
        for r in range(N):
            padding_mask[r, :patch_mask[r]] = torch.tensor(float('-inf'))
        return padding_mask