import pandas as pd
from Battery.Models.Tr_model import model
from torch.utils.data import DataLoader
import torch
from CustomDataset import CustomDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Battery.utils import get_mae, get_rmse, get_r2, get_max_err

torch.set_printoptions(precision=16)
test_data = pd.read_csv("../PostProcess/11column4sample/test_data.csv")
test_data = test_data[:]

# model Setting
device = 'cuda'
result_type = "current"
weight_decay = 0.0
sequence_length = 4
# model setting

processed_data = [test_data]

saved_model_file = 'Tr_10e_4sl_current'

model.load_state_dict(torch.load(f'../Models/savedParams/{saved_model_file}.pth', weights_only=True,
                                 map_location=torch.device(device)))
model = model.to(device)

train_dataset = CustomDataset(processed_data, result_type, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
y_true = train_dataset.get_y_true()
y_pred = []

print(train_dataset.__len__())

for i, data in enumerate(tqdm(train_loader, desc="Processing batches", unit="batch")):
    current_sequence = data["data"].float()
    for sequence in range(1, min(len(y_pred) + 1, 5)):  # y_pred 길이와 4 중 작은 값까지만 순회
        current_sequence[-1, -sequence, -1] = y_pred[-sequence]
    X = current_sequence
    X = X.to(device)

    pred = model(X).item()
    y_pred.append(pred)

y_true = np.array(y_true) if isinstance(y_true, list) else y_true
y_pred = [pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred) for pred in y_pred]
y_pred = np.array(y_pred)
print(y_true.__len__(), y_pred.__len__())

rmse = get_rmse(y_true, y_pred)
mae = get_mae(y_true, y_pred)
max_err = get_max_err(y_true, y_pred)
r2 = get_r2(y_true, y_pred)

print("Metrics:")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Max Error: {max_err}")
print(f"R² Coefficient: {r2}")

y_true = np.array(y_true).flatten()
y_pred = np.array(y_pred).flatten()

# draw graph
plt.figure(figsize=(10, 6))
plt.plot(y_true, label='True values', color='blue', marker='o')
plt.plot(y_pred, label='Predicted values', color='red', marker='x')

plt.title('True vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.legend()
plt.show()
