import numpy as np
import torch
import torch.nn.functional as F

class PolynomialKernel(torch.nn.Module):
    def __init__(self, c=0.0, degree=3):
        super(PolynomialKernel, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.tensor(c), requires_grad=False)
        self.degree = torch.nn.parameter.Parameter(torch.tensor(degree), requires_grad=False)

    def forward(self, x, w, b):
        w = w.view(w.size(0), -1).t()
        out = (x.matmul(w) + self.c) ** self.degree
        return out

class GaussianKernel(torch.nn.Module):
    def __init__(self, gamma=0.5):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(gamma), requires_grad=False)

    def forward(self, x, w, b):
        w = w.view(w.size(0), -1)
        l2_dist = torch.cdist(x, w, p=2) ** 2
        out = torch.exp(-self.gamma * l2_dist)
        return out


class SigmoidKernel(torch.nn.Module):
    def __init__(self):
        super(SigmoidKernel, self).__init__()

    def forward(self, x, w, b):
        w = w.view(w.size(0), -1).t()
        out = x.matmul(w).tanh()
        return out

