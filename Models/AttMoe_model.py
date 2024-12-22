from torch import nn
from mixture_of_experts import MoE


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


class AttMoE(nn.Module):
    def __init__(self, feature_size, hidden_dim, nhead=4, dropout_att=0.,
                 num_experts=8):
        super(AttMoE, self).__init__()
        self.feature_size, self.hidden_dim = feature_size, hidden_dim
        self.cell = Attention(feature_size=feature_size, hidden_dim=hidden_dim, nhead=nhead, dropout=dropout_att)
        self.linear = nn.Linear(hidden_dim, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        experts = nn.Linear(hidden_dim, hidden_dim)

        self.moe = MoE(dim=hidden_dim,
                       num_experts=num_experts,
                       experts=experts)
        self.moe = self.moe

    def forward(self, x):
        out = self.cell(x)
        out, _ = self.moe(out)
        out = out.permute(0, 2, 1)
        out = self.global_avg_pool(out)
        out = out.squeeze(-1)
        out = self.linear(out)
        out = out.squeeze(-1)
        return out


# model Setting #
input_size = 10
# Attention
att_hidden_dim = 128
nhead = 4
# Attention

# MoE
num_experts = 12
# biLSTM
# model setting

model = AttMoE(input_size, hidden_dim=att_hidden_dim, nhead=nhead, num_experts=8)
