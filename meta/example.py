import argparse
import random

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from meta.algrithm import MAML


def compute_loss(model):
    pass


model = MyModel()
maml = MAML(model, lr=0.1)
opt = torch.optim.SGD(maml.parameters(), lr=0.001)   # change it
for iteration in range(10):
    opt.zero_grad()
    task_model = maml.clone()  # torch.clone() for nn.Modules
    adaptation_loss = compute_loss(task_model)
    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place, i.e. one inner step
    evaluation_loss = compute_loss(task_model)
    evaluation_loss.backward()  # gradients w.r.t. maml.parameters()
    opt.step()
