import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, normalizer):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = nn.Linear(hidden_dim, num_outputs)
        self.normalizer = normalizer
        self.type = 'deterministic'

    def forward(self, x):
        #x = self.normalizer(x)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = torch.tanh(self.affine3(x))
        return x


class TD3Value(nn.Module):
    def __init__(self, num_inputs, hidden_dim, normalizer):
        super(TD3Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = nn.Linear(hidden_dim, 1)

        self.affine4 = nn.Linear(num_inputs, hidden_dim)
        self.affine5 = nn.Linear(hidden_dim, hidden_dim)
        self.affine6 = nn.Linear(hidden_dim, 1)

        """
        self.value_head = nn.Linear(hidden_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        """

        self.normalizer = normalizer

    def forward(self, x, u):
        #x = self.normalizer(x)
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.affine1(xu))
        x1 = F.relu(self.affine2(x1))
        x1 = self.affine3(x1)

        x2 = F.relu(self.affine4(xu))
        x2 = F.relu(self.affine5(x2))
        x2 = self.affine6(x2)
        return x1, x2

    def Q1(self, x, u):
        #x = self.normalizer(x)
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.affine1(xu))
        x1 = F.relu(self.affine2(x1))
        x1 = self.affine3(x1)
        return x1



class StochasticPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, normalizer):
        super(StochasticPolicy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, hidden_dim)

        self.action_mean = nn.Linear(hidden_dim, num_outputs)
        #self.action_mean.weight.data.mul_(0.1)
        #self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.normalizer = normalizer
        self.type = 'stochastic'

    def forward(self, x):
        #x = self.normalizer(x)
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = torch.tanh(self.action_mean(x))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std

    def entropy(self, x):
        mean, log_std, std = self.forward(x)
        m = Normal(mean, std)
        return m.entropy().mean()
    

class Value(nn.Module):
    def __init__(self, num_inputs, hidden_dim, normalizer):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.normalizer = normalizer

    def forward(self, x):
        #x = self.normalizer(x)
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
