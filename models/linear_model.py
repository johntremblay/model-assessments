"""
A linear model class for deep learning
"""

import math
import torch
import logging


class LinearModel:
    def __init__(self, size1, size2, lr):
        self.weights = torch.randn(size1, size2) / math.sqrt(size1)
        self.bias = torch.zeros(size2, requires_grad=True)
        self.weights.requires_grad_()
        self.lr = lr

    @staticmethod
    def log_softmax(x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)

    def output(self, output_activation, x):
        return output_activation(x @ self.weights + self.bias)

    @staticmethod
    def nll(input, target):
        return -input[range(target.shape[0]), target].mean()

    @staticmethod
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    def evaluate(self, input, targets):
        _loss = self.nll(self.output(self.log_softmax, input), targets)
        _acc = self.accuracy(self.output(self.log_softmax, input), targets)
        return _loss.item(), _acc.item()

    def training(self, train_input, train_targets, eval_input, eval_targets, epochs, batch_size, verbose=True):
        n, c = train_input.shape
        for epoch in range(epochs):
            for i in range((n - 1) // batch_size + 1):
                start_i = i * batch_size
                end_i = start_i + batch_size
                xb = train_input[start_i:end_i]
                yb = train_targets[start_i:end_i]
                pred = self.output(self.log_softmax, xb)
                loss = self.nll(pred, yb)

                loss.backward()
                with torch.no_grad():
                    self.weights -= self.weights.grad * self.lr
                    self.bias -= self.bias.grad * self.lr
                    self.weights.grad.zero_()
                    self.bias.grad.zero_()

            if verbose and (epoch % 10 == 0):
                print(f'train set {self.evaluate(train_input, train_targets)} at epoch {epoch}')
                print(f'eval set {self.evaluate(eval_input, eval_targets)} at epoch {epoch}')


