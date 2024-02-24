import torch
import torchvision
import torch.nn as nn

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

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

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int = 2, d_hid: int = 128, num_class: int = 1,
                 nlayers: int = 16, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model*ntoken, num_class)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = output.permute(1, 0, 2)
        output = output.reshape(output.size(0), -1)
        output = self.linear(output)
        return output


# Vision Transformer: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# try pretrain, same link above
class VT_CNN(nn.Module):
    def __init__(self, num_class=2):
        super(VT_CNN, self).__init__()

        # =======
        # Unet
        self.conv_init = nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.VisionTransformer = torchvision.models.VisionTransformer(
            image_size = 100,
            patch_size = 10,
            num_layers = 16,
            num_heads = 5,
            hidden_dim = 250,
            mlp_dim = 520
        )
        self.ln_init = nn.LayerNorm(1000)
        self.fc = nn.Linear(1000, num_class)
        self.fc_res = nn.Linear(3*100*100, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Input scale: (0, 255)
        Output scale: (0, 255)
        """
        
        x = self.conv_init(x)
        x_ = x.view(x.size(0), -x.size(1)*x.size(2)*x.size(3))
        x = self.VisionTransformer(x)
        x = self.ln_init(x)
        x = self.relu(x)        
        res = self.fc_res(x_)
        x = self.fc(x)
        x = x + res

        return x