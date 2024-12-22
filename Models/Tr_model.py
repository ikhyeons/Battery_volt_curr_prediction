import torch
import torch.nn as nn
from mixture_of_experts import MoE
import math


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
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim=1):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 마지막 시점의 출력을 사용
        output = self.fc_out(x)
        output = output.squeeze(-1)  # (16, 1) -> (16)
        return output


# model Setting #
input_dim = 10
model_dim = 128
num_heads = 4
num_layers = 6
# model setting

model = Transformer(input_dim, model_dim, num_heads, num_layers)
