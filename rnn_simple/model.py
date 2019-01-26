from __future__ import print_function, absolute_import

import os
import sys

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from src import metrics
from src.models.nn_model import NNModel
from src.util.torch import to_torch

class Gen:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = len(X)
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        start = self.i
        self.i += 20
        if self.Y is None:
            return (self.X[start: self.i], None)
        else:
            return (self.X[start: self.i], self.Y[start: self.i])
    
    def get_Y(self):
        return self.Y


class Network(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.device = torch.device(device)
        self.input_size = 1
        self.hidden_size = 3
        self.output_size = 1

        self.w_ih = torch.nn.Parameter(data=torch.Tensor(self.input_size, self.hidden_size), requires_grad=True)
        self.w_hh = torch.nn.Parameter(data=torch.Tensor(self.hidden_size, self.hidden_size), requires_grad=True)
        self.w_ho = torch.nn.Parameter(data=torch.Tensor(self.hidden_size, self.output_size), requires_grad=True)
        
        for p in [self.w_ih, self.w_hh, self.w_ho]:
            nn.init.normal(p, 0.0, 0.4)

        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=config.lr, momentum=config.momentum, eps=1e-6)

    def forward(self, x_t, y_t):
        loss_t = 0
        pred_ts = []
        for x_ti, y_ti in zip(x_t, y_t):
            context_state = Variable(torch.zeros((1, self.hidden_size), dtype=torch.float32), requires_grad=True).to(self.device)
            for x_iti in x_ti:
                context_state = torch.tanh(x_iti * self.w_ih + context_state.mm(self.w_hh))
            pred_ti = context_state.mm(self.w_ho)
            pred_ts.append(pred_ti)
        pred_t = torch.cat(pred_ts)[:, 0]
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
        

class Model(NNModel):
    def init_model(self):
        network = Network(self.device, self.config)
        self.set_network(network)
    
    def generate_sequences(self, size):
        starts = np.random.randint(-50, 200, size=size)
        steps = np.random.randint(1, 20, size=size)
        num_steps = np.random.randint(2, 20, size=size)
        X, Y = [], []
        for start, step, num_step in zip(starts, steps, num_steps):
            X.append(np.array([(start + step * i) for i in range(num_step)], dtype=np.float32))
            Y.append(np.float32(start + step * num_step))
        return Gen(X, np.array(Y))
    
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