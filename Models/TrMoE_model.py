import torch
import torch.nn as nn
from mixture_of_experts import MoE
import math


class TrMoE(nn.Module):
    def __init__(self, input_dim, tr_model_dim, num_heads, num_layers, moe_model_dim, num_experts):
        super(TrMoE, self).__init__()
        self.model_name = 'TrMoE'
        experts = nn.Linear(tr_model_dim, moe_model_dim)
        self.tr = Transformer(input_dim, tr_model_dim, num_heads, num_layers)
        self.moe = MoE(dim=moe_model_dim,
                       num_experts=num_experts,
                       experts=experts)
        self.linear1 = nn.Linear(moe_model_dim, 1)
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.tr(x)
        x, _ = self.moe(x)
        x = self.linear1(x)
        x = x.squeeze(-1)
        x = self.linear2(x)
        out = x.squeeze(-1)
        return out



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)


class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer_encoder(x)
        return out


# model Setting #
# tr
input_dim = 10
num_heads = 4
num_layers = 4
tr_model_dim = 128
# tr
# moe
moe_model_dim = 128
num_experts = 12
# moe
# model setting


model = TrMoE(input_dim, tr_model_dim, num_heads, num_layers, moe_model_dim, num_experts)
