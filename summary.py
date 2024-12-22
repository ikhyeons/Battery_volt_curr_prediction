from common_utils import print_model_parameters
from Battery.Models.Bi_LSTM_model import BiLSTM

# AttMoE
# nhead = 4
# weight_decay = 0.0
# dropout_att = 0.0
# hidden_dim = 256
# lr = 0.0001
# seed = 0
# num_experts = 16
# device = 'cuda'
# feature_size = 10  # 각 피쳐들의 수 (전압, 전류, 압력 포함 총 10개)
# sequence_length = 4  # 64개의 과거데이터를 입력
# model = AttMoe_model.AttMoE(
#     feature_size=feature_size,
#     hidden_dim=hidden_dim,
#     nhead=nhead,
#     dropout_att=dropout_att,
#     num_experts=num_experts,
#     device=device)

#Bi_LSTM
input_size = 10
hidden_size = 128
num_layers = 2
bidirectional = True
weight_decay = 0.0
output_size = 1
sequence_length = 4
model = BiLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    bidirectional=bidirectional
    )

print_model_parameters(model)
