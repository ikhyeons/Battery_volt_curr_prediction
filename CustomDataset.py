import pandas as pd
from torch.utils.data import Dataset
import torch
from Battery.secrets import columns_except_volt, columns_except_current
from Battery.secrets import volt_result_set, current_result_set
from Battery.secrets import volt_column, curr_column
import numpy as np


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def create_all_x_y(df, targettype):
    result = df.copy()
    if targettype == "volt":
        result = result[volt_result_set]
    else:
        result = result[current_result_set]
    return result


class CustomDataset(Dataset):
    def __init__(self, datalist, targettype, sequence_length=64):
        self.sequence_length = sequence_length
        self.datalist = []
        for data in datalist:
            self.datalist.append(create_all_x_y(data, targettype))
        self.X = []
        self.y = []

        for data in self.datalist:
            for i in range(len(data) - sequence_length):
                progress = (i / (len(data) - sequence_length)) * 100
                print(f"\rProgress: {progress:.3f}% complete", end="")
                # X는 sequence_length 크기 만큼의 데이터
                if targettype == "volt":
                    self.X.append(data.iloc[i:i + sequence_length][columns_except_volt].values)  # 전압빼고 다 학습
                    self.y.append(data.iloc[i + sequence_length-1][volt_column])  # 정답 전압
                else:
                    self.X.append(data.iloc[i:i + sequence_length][columns_except_current].values)  # 전류빼고 다 학습
                    self.y.append(data.iloc[i + sequence_length-1][curr_column])  # 정답 전류
            print()  # Progress 출력 후 줄바꿈

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx]
        y = self.y[idx]

        return {
            'data': torch.from_numpy(data),
            'y': y
        }

    def get_test_init(self):
        y_pred = []
        y_true = []
        for i in range(0, self.sequence_length):
            y_pred.append(self.__getitem__(i)['y'])
            y_true.append(self.__getitem__(i)['y'])
        return y_pred, y_true
