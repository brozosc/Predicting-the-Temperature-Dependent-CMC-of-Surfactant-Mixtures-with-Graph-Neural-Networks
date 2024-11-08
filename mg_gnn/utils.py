# -*- coding: utf-8 -*-
"""
@author: BrozosCh
"""
import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae



def error_metrics(y_true, y_pred):
    r_mae = mae(y_true,y_pred)
    r_rmse = np.sqrt(mse(y_true,y_pred))
    y_true_non_zero = [x for x in y_true if x != 0]
    y_pred_non_zero = [x for i, x in enumerate(y_pred) if y_true[i] != 0]
    r_mape = mape(y_true_non_zero, y_pred_non_zero)
    return r_mae, r_rmse, r_mape

# adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose = False,delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.min_validation_loss = np.Inf
        self.delta = delta


    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    def reset_parameters(self):
        self.counter = 0
        self.best_score = None
        self.min_validation_loss = np.Inf
        self.early_stop = False