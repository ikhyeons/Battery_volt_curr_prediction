import pandas as pd
from AttMoe_model import AttMoE
from Battery.utils import setup_seed
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from Battery.CustomDataset import CustomDataset
from torch.optim.lr_scheduler import StepLR
from Battery.secrets import train_file_path_list

# hyper params Setting #
EPOCH = 3
lr = 0.0001
seed = 0
# hyper params Setting #

# model Setting #
device = 'cuda'
nhead = 4
weight_decay = 0.0
dropout_att = 0.0
hidden_dim = 256
num_experts = 12
feature_size = 10
sequence_length = 4

result_type = "volt"
# model Setting #

# train file list
filePaths = train_file_path_list
# set seed
setup_seed(seed)
# get model
model = AttMoE(feature_size=feature_size, hidden_dim=hidden_dim,
    nhead=nhead, dropout_att=dropout_att, num_experts=num_experts, device=device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.MSELoss()

processed_data = []

# get files
for file_path in filePaths:
    print(f"Read on file: {file_path}")
    processed_data.append(pd.read_csv(f"../1PostProcess/11column/{file_path}.csv"))

train_dataset = CustomDataset(processed_data, result_type, sequence_length)
print(train_dataset.__len__())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(EPOCH):
    print(f'Epoch {epoch + 1}', end=" ")
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        X = data["data"].float()
        X = X.to(device)
        y = data["y"].float()
        y = y.to(device)
        outputs = model(X)
        outputs = outputs
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Loss: {epoch_loss:.8f}')

torch.save(model.state_dict(), f'savedModels/AttMoE_{EPOCH}e_4ss_{result_type}.pth')
