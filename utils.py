import os
import numpy as np
import random
import torch
from sklearn.metrics import mean_absolute_error, r2_score, max_error, mean_squared_error

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def get_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def get_max_err(y_true, y_pred):
    return max_error(y_true, y_pred)


def get_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)