import numpy as np
import torch
import torch.nn.functional as F

class PolynomialKernel(torch.nn.Module):
    def __init__(self, c=1.0, degree=2):
        super(PolynomialKernel, self).__init__()
        self.c = torch.nn.parameter.Parameter(torch.tensor(c), requires_grad=False)
        self.degree = torch.nn.parameter.Parameter(torch.tensor(degree), requires_grad=False)

    def forward(self, x, w):
        w = w.view(w.size(0), -1).t()
        out = (x.matmul(w) + self.c) ** self.degree
        return out

class GaussianKernel(torch.nn.Module):
    def __init__(self, gamma=0.5):
        super(GaussianKernel, self).__init__()
        self.gamma = torch.nn.parameter.Parameter(torch.tensor(gamma), requires_grad=False)

    def forward(self, x, w):
		# Calculate L2-norm
        l2 = x.unsqueeze(3) - w.view(1, 1, -1, w.size(0))
        l2 = torch.sum(l2 ** 2, 2)

        out = torch.exp(-self.gamma * l2)
        return out

class SigmoidKernel(torch.nn.Module):
    def __init__(self):
        super(SigmoidKernel, self).__init__()

    def forward(self, x, w):
        w = w.view(w.size(0), -1).t()
        out = x.matmul(w).tanh()
        return out

