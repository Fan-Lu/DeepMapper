# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:13:56 2024

@author: fanlu
"""

import numpy as np
import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F


class SingleHead(nn.Module):
    """
    Single head self attention

    """

    def __init__(self, n_embed, seq_len, head_size, dropout):
        """


        Parameters
        ----------
        n_embed : TYPE
            DESCRIPTION.
        seq_len : TYPE
            DESCRIPTION.
        head_size : TYPE
            DESCRIPTION.
        dropout : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # torch.ones(seq_len, seq_len) create a seq_len x seq_len matrix that contains all ones
        # torch.tril(A) return the lower triangular part the matrix A, other parts are set to zero
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=False):
        """


        Parameters
        ----------
        x : TYPE, torch.tensor (torch.float32)
            DESCRIPTION. Embedding matrix that contains a sequence of word embeddings
            x.shape = B x T x C,
                B is the batch size
                T is the sequence lence
                C is the dimension of embedding (number of features in each embedding)
        mask : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        B, T, C = x.shape
        # shape of k is B x T x T
        k = self.key(x)
        # shape of q is B x T x T
        q = self.query(x)
        # shape of v is B x T x T
        v = self.value(x)

        # compute attention scores (affinities)
        # dimension of 1 and 2 are swapped
        attention_score = q @ k.transpose(1, 2) / math.sqrt(C)

        if mask:
            # decoder masked self attention
            attention_score = attention_score.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        attention_score = F.softmax(attention_score, dim=-1)  # (B, T, T)
        attention_score = self.dropout(attention_score)
        # perform the weighted aggregation of the values
        out = attention_score @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHead(nn.Module):

    def __init__(self, n_embed, seq_len, num_heads, head_size, dropout):
        super().__init__()

        # create a list of single head
        self.heads = nn.ModuleList([
            SingleHead(n_embed, seq_len, head_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=False):
        # TODO: Dimension is not correct here
        out = torch.cat([h(x, mask) for h in self.heads], dim=2)
        out = self.dropout(self.proj(out))
        return out


def fully_connected(n_embed, fc_dim, dropout):
    return nn.Sequential(
        nn.Linear(n_embed, fc_dim),
        nn.ReLU(),
        nn.Linear(fc_dim, n_embed),
        nn.Dropout(dropout),
    )


def get_angles(pos, k, d):
    # k: [0, d)
    i = k // 2
    angles = pos / (10000 ** (2 * i / d))
    return angles

def positional_encoding(positions, d):
    """
    Precompute a matrix with all positional encodings
    Arguments:
        positions (int) : maximum number of positions to be encoded
        d (int) : Encoding size

    Returns:
        pos_encoding: (1, position, d_model), A matrix with the positonal encodings

    """
    angle_rads = get_angles(np.arange(positions).reshape(positions, 1), np.arange(d), d)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis,]
    return torch.tensor(pos_encoding, dtype=torch.float32)

class EncoderLayer(nn.Module):

    def __init__(self, n_embed, seq_len, num_heads, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.attn = MultiHead(n_embed, seq_len, num_heads, head_size, dropout)
        self.ffwd = fully_connected(n_embed, 4 * n_embed, dropout)
        self.ln1 = nn.LayerNorm(normalized_shape=n_embed)
        self.ln2 = nn.LayerNorm(normalized_shape=n_embed)

    def forward(self, x):
        # residual connection
        x = x + self.ln1(self.attn(x))
        x = x + self.ln2(self.ffwd(x))
        return x


class Encoder(nn.Module):

    def __init__(self, n_layers, n_embed, seq_len, num_heads, dropout, vocab_size):
        super().__init__()
        self.n_embed = n_embed
        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embeddding = nn.Embedding(vocab_size, n_embed)
        # self.pos_encoding = positional_encoding(seq_len, n_embed)
        self.att_layers = nn.ModuleList([EncoderLayer(n_embed, seq_len, num_heads, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """


        Parameters
        ----------
        x : TYPE  shape (B, T)
            DESCRIPTION.

        Returns
        -------
        None.

        """
        seq_len = x.shape[1]
        x = self.embeddding(x)  # shape (B, T, C), C = n_embed
        x *= self.n_embed ** 0.5
        x = x + self.pos_encoding[:, :seq_len, :]  # shape (B, T, C)
        x = self.dropout(x)
        for layer in self.att_layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, n_embed, seq_len, num_heads, dropout):
        super().__init__()

        head_size = n_embed // num_heads
        self.masked_attn = MultiHead(n_embed, seq_len, num_heads, head_size, dropout)
        self.attn = MultiHead(n_embed, seq_len, num_heads, head_size, dropout)
        self.ffwd = fully_connected(n_embed, 4 * n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)

    def forward(self, x, encoder_output=None):
        x = self.ln1(x + self.masked_attn(x, mask=True))
        # x = self.ln2(x + self.attn(encoder_output))
        # x = self.ln2(x)
        x = self.ln3(x + self.ffwd(x))
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, n_embed, seq_len, num_heads, dropout, vocab_size):
        super().__init__()
        self.n_embed = n_embed
        self.embeddding = nn.Embedding(vocab_size, n_embed)
        # self.pos_encoding = positional_encoding(seq_len, n_embed)
        self.att_layers = nn.ModuleList([DecoderLayer(n_embed, seq_len, num_heads, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        # x (B, T)
        seq_len = x.shape[1]
        x = self.embeddding(x)  # (B, T, C)
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x *= self.n_embed ** 0.5
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.att_layers:
            x = layer(x, encoder_output)
        return x


class Tranformer(nn.Module):

    def __init__(self, n_layers, n_embed, seq_len, num_heads, dropout, vocab_size):
        super().__init__()
        self.encoder = Encoder(n_layers, n_embed, seq_len, num_heads, dropout, vocab_size)
        self.decoder = Decoder(n_layers, n_embed, seq_len, num_heads, dropout, vocab_size)

        # self.lm_head = nn.Sequential(
        #     [nn.Linear(n_embed, vocab_size),
        #     F.softmax(dim=-1)]
        #     )
        self.lm_head = nn.Sequential(
            nn.Linear(n_embed, vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, input_sequece, output_sequence):
        """
        Forward pass for the entire Transformer

        input_sequece -- Tensor of shape (B, input_sequence_len)
                            An array of indexes of words in the input sequence

        output_sequence -- Tensor of shape (B, output_sequence_len)
                            An array of indexes of the words in the ouput sequence

        Reurns:
            final_output -- probability of the next token


        """

        encoder_output = self.encoder(input_sequece)
        decoder_output = self.decoder(output_sequence, encoder_output)
        x = self.lm_head(decoder_output)
        return x


# define the NN architecture
class AttentionAE(nn.Module):
    def __init__(self, h_dim=4, seq_len=8, dropout=0.5):
        super(AttentionAE, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # conv layer (depth from 8 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        # conv layer (depth from 4 --> 1), 3x3 kernels
        # self.conv3 = nn.Conv2d(4, 1, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        self.enc_fc1 = nn.Linear(4 * 32 * 32, 128)
        self.enc_fc2 = nn.Linear(128, h_dim)
        # self.enc_fc3 = nn.Linear(64, h_dim)

        self.dec_fc1 = nn.Linear(h_dim, 128)
        self.dec_fc2 = nn.Linear(128, 4 * 32 * 32)

        self.A_fc1 = nn.Linear(h_dim, 16)
        self.A_fc2 = nn.Linear(16, 3)
        # self.A_fc3 = nn.Linear(64, 32)
        # self.A_fc4 = nn.Linear(32, 3)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1 = nn.ConvTranspose2d(1, 4, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_b1 = nn.BatchNorm2d(16)
        self.t_conv2 = nn.ConvTranspose2d(32, 3, 2, stride=2)
        self.t_b2 = nn.BatchNorm2d(3)

        # sampling time
        # self.sample_time = 0.1001
        self.sample_time = 0.2

        self.Kh = 0.5
        self.Ki = 0.3
        self.Kp = 0.1

        self.ks_omega = nn.Parameter(torch.from_numpy(np.array([0.5, 0.3, 0.1])).float(), requires_grad=True)

        self.Amat_masked = torch.zeros((3, 4, 4), requires_grad=False)
        self.Amat_masked[0][0][0] = -1.0
        self.Amat_masked[0][1][0] =  1.0
        self.Amat_masked[1][1][1] = -1.0
        self.Amat_masked[1][2][1] =  1.0
        self.Amat_masked[2][2][2] = -1.0
        self.Amat_masked[2][3][2] =  1.0

        self.softmax = nn.Softmax(dim=1)

        self.mina = torch.from_numpy(np.array([0.1, 0.1, 0.1]))
        self.maxa = torch.from_numpy(np.array([0.9, 0.9, 0.9]))

        self.n_embed = h_dim
        self.seq_len = seq_len
        self.num_heads = 1
        n_layers = 1

        self.dropout = nn.Dropout(dropout)
        self.att_layers = nn.ModuleList([DecoderLayer(self.n_embed, self.seq_len, self.num_heads, dropout) for _ in range(n_layers)])

    def encoder(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        bs = x.shape[0]
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # compressed representation
        # add second hidden layer
        # 1 X 16 X 64 X 64
        self.feat_layer_1 = x
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        self.feat_layer_2 = x
        # add third hidden layer
        # 1 X 4 X 32 X 32
        x = x.view(bs, -1)
        x = F.relu(self.enc_fc1(x))
        # x = F.softmax(self.enc_fc2(x), dim=1)
        x = F.sigmoid(self.enc_fc2(x))
        # x = F.relu(self.enc_fc2(x))

        return x

    def decoder(self, z):
        ## decode ##
        bs = z.shape[0]
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = z.view(bs, 4, 32, 32)
        z = torch.concat([z, self.feat_layer_2.data], dim=1)
        # add transpose conv layers, with relu activation function
        z = F.relu(self.t_conv1(z))
        # add transpose conv layers, with relu activation function
        z = torch.concat([z, self.feat_layer_1.data], dim=1)
        z = F.sigmoid(self.t_conv2(z))

        return z

    def attention(self, z):
        # z shape (B, T, C)
        # seq_len = z.shape[1]
        # Scale embedding by multiplying it by the square root of the embedding dimension
        # z *= self.n_embed ** 0.5
        z = z + 2.0 * positional_encoding(z.shape[1], 4)
        z = self.dropout(z)
        for layer in self.att_layers:
            z = layer(z)
        ks = F.relu(self.A_fc1(z))
        ks = F.tanh(self.A_fc2(ks))
        # ks = F.sigmoid(self.A_fc2(ks))
        ks_omega = F.sigmoid(self.ks_omega)

        return ks + ks_omega

    def shift(self, z, ks, time_dif=7200 * 5):
        # ks = torch.clip(ks, self.mina, self.maxa).float()
        ks = torch.clip(ks, 0, 1).float()
        self.Kh, self.Ki, self.Kp = ks.cpu().data.squeeze().numpy()
        Amat = torch.tensordot(ks.reshape(1, 1, -1), self.Amat_masked, dims=[[2], [0]]).squeeze()
        Az = torch.matmul(Amat, z)
        # default sample time is 7200 seconds
        z_next = z + Az * self.sample_time * (time_dif / 7200.0)
        return z_next

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat
