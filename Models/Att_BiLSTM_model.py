import torch
import torch.nn as nn


class AttBiLSTM(nn.Module):
    def __init__(self, input_size, att_hidden_dim, nhead, lstm_hidden_size, num_layers, output_size):
        super(AttBiLSTM, self).__init__()
        self.attention = Attention(input_size, att_hidden_dim, nhead)

        self.hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(att_hidden_dim, lstm_hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        x = self.attention(x)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        out = x.squeeze(1)

        return out


class Attention(nn.Module):
    def __init__(self, feature_size, hidden_dim, nhead=4, dropout=0.0):
        super(Attention, self).__init__()
        self.query = nn.Linear(feature_size, hidden_dim)
        self.key = nn.Linear(feature_size, hidden_dim)
        self.value = nn.Linear(feature_size, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, x):
        query, key, value = self.query(x), self.key(x), self.value(x)
        out, _ = self.attn(query, key, value)
        return out


# model Setting #
input_size = 10
# Attention
att_hidden_dim = 128
# Attention

# biLSTM
lstm_hidden_size = 128
nhead = 4
num_layers = 2
output_size = 1
# biLSTM
# model setting

model = AttBiLSTM(input_size, att_hidden_dim, nhead, lstm_hidden_size, num_layers, output_size)


