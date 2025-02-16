import torch.nn as nn
from models.TransBTS.IntmdSequential import IntermediateSequential
from torch.nn import init
from scipy import linalg
import torch.nn.functional as F
import torch
import math

##AugTransformer
class Circulant_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_cir=4):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_cir = num_cir

        self.min_features = min(in_features, out_features)
        self.num_cir_in = self.num_cir * in_features // self.min_features
        self.num_cir_out = self.num_cir * out_features // self.min_features
        # print('mode:',mode,'self.num_cir_out',self.num_cir_out,'self.num_cir_in',self.num_cir_in)

        assert self.min_features % self.num_cir == 0
        self.fea_block = self.min_features // self.num_cir

        self.weight_list = nn.Parameter(torch.Tensor(self.num_cir_out * self.num_cir_in * self.fea_block))

        index_matrix = torch.from_numpy(linalg.circulant(range(self.fea_block))).long()

        index_list = [[] for iblock in range(self.num_cir_out)]
        for iblock in range(self.num_cir_out):
            for jblock in range(self.num_cir_in):
                index_list[iblock].append(index_matrix + (iblock * self.num_cir_in + jblock) * self.fea_block)
        for iblock in range(self.num_cir_out):
            index_list[iblock] = torch.cat(index_list[iblock], dim=1)
        # print('weight_list',weight_list)
        self.register_buffer('index_matrix', torch.cat(index_list, dim=0))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_list.view(self.num_cir_out, self.num_cir_in, self.fea_block), a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight_list[self.index_matrix]
        return F.linear(input, weight, self.bias)

#Transformer Layer
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

##Augmented shortcuts
class Residual1(nn.Module):
    def __init__(self, fn, dim, num_cir=4):
        super().__init__()
        self.fn = fn
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)  ##ldw add##
        self.augs_attn = Circulant_Linear(in_features=dim, out_features=dim, bias=True, num_cir=num_cir)


    def forward(self, x):
        res = x
        res = res.permute(0, 2, 1)
        res = self.conv(res)
        res = res.permute(0, 2, 1)
        x = res + x
        return self.fn(x) + x + self.augs_attn(x)

class Residual2(nn.Module):
    def __init__(self, fn, dim,num_cir=4):
        super().__init__()
        self.fn = fn
        self.augs = Circulant_Linear(in_features=dim, out_features=dim, bias=True, num_cir=num_cir)

    def forward(self, x):
        return self.fn(x) + x + self.augs(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual1(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        ),
                        dim
                    ),
                    Residual2(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate)),
                        dim
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)
