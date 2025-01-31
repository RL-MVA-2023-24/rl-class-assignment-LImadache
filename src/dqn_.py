import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, neurons, device='cpu'):
        self.input_size = 6
        self.output_size = 4
        self.hidden_size = neurons

        super(DQN, self).__init__()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, device=device)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, device=device)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size, device=device)

        self.bn = nn.BatchNorm1d(self.input_size)

    def forward(self, x):
        # x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc3(x)
        # x = F.relu(x)
        return x