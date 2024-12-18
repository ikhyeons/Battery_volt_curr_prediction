import pandas as pd
from Bi_LSTM_model import BiLSTM
from torch.utils.data import DataLoader
import torch
from Battery.CustomDataset import CustomDataset
from Battery.utils import get_mae, get_r2, get_rmse, get_max_err
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_printoptions(precision=16)
test_data = pd.read_csv("../1PostProcess/11column/test_data.csv")
test_data = test_data[:-20000]

processed_data = []
processed_data.append(test_data)

# setting #
device = 'cuda'
input_size = 10
hidden_size = 256
bidirectional = True
weight_decay = 0.0
output_size = 1
sequence_length = 4
num_layers = 2

saved_model_file = "BiLSTM_10e_4ss_curr"
result_type = "volt"
# setting #

model = BiLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size,
    bidirectional=bidirectional
    )

model.load_state_dict(torch.load(f'savedModels/{saved_model_file}.pth', weights_only=True))
model = model.to(device)

train_dataset = CustomDataset(processed_data, result_type, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
y_pred, y_true = train_dataset.get_test_init()

print(train_dataset.__len__())

for i, data in enumerate(tqdm(train_loader, desc="Processing batches", unit="batch")):
    if i < sequence_length:
        continue
    X = data["data"].float()

    last_pred_volt = y_pred[-1]
    X[-1, -1, -1] = last_pred_volt

    X = X.to(device)
    y_true.append(data['y'].float().item())
    y_pred.append(model(X).item())

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
