from __future__ import print_function, absolute_import

import os
import sys

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from src import metrics
from src.generators.matrix import MatrixGen
from src.models.nn_model import NNModel
from src.util.torch import to_torch


class Network(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.device = torch.device(device)
        hidden_size = 10
        self.input = nn.Linear(1, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, dropout=config.dropout)
        self.output = nn.Linear(hidden_size, 1)
        
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=config.lr, momentum=config.momentum, eps=1e-6)

    def forward(self, x_t, y_t):
        pred_ts = []
        inp = self.input(x_t.unsqueeze(2))
        output, (h, c) = self.lstm(inp)
        pred_t = self.output(output).squeeze()
        if y_t is None:
            return None, {
                'x': x_t,
                'y': y_t,
                'pred': pred_t
            }
        loss_t = self.loss(y_t, pred_t)
        return loss_t, {
            'x': x_t,
            'y': y_t,
            'loss': loss_t,
            'pred': pred_t
        }

def sine_2(X, signal_freq=60.):
    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def sample(sample_size):
    random_offset = np.random.randint(0, sample_size)
    X = np.arange(sample_size)
    Y = noisy(sine_2(X + random_offset))
    return Y

class Model(NNModel):
    @classmethod
    def get_params(self, config):
        return {
            'momentum': np.random.choice([0.9, 0.95]),
            'lr': np.random.choice([1e-2, 12e-3, 1e-4]),
            'dropout': np.random.choice([0, 0.05, 0.2])
        }

    def init_model(self):
        network = Network(self.device, self.config)
        self.set_network(network)
    
    def generate_sequences(self, size):
        sine = np.array([sample(50) for _ in range(size)], dtype=np.float32)
        return MatrixGen(sine[:, :-1], sine[:, 1:])
    
    def get_train_val_data(self):
        return self.generate_sequences(500), self.generate_sequences(100)

    def get_pred_data(self, path):
        return self.generate_sequences(100)
    
    def get_test_data(self):
        return self.generate_sequences(100)
    
    def train_metrics(self, y_true, pred):
        return self.eval_metrics(y_true, pred)
    
    def eval_metrics(self, y_true, pred):
        return {
            'loss': pred['loss'],
            'mse': metrics.mse(y_true, pred['pred'])
        }