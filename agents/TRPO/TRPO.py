import scipy.optimize
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence

from agents.TRPO.trpo import trpo_step
from agents.TRPO.utils import *

####################3
from models.agent import StochasticPolicy, Value
from utils import device
####################

torch.set_default_tensor_type('torch.DoubleTensor')

def update_params(batch, policy_net, value_net, gamma, tau, l2_reg, max_kl, damping, entropy_coef):
    rewards = torch.Tensor(batch.reward)
    
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(batch.action)
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_prob = policy_net(Variable(states))
    fixed_log_prob = log_density(Variable(actions), action_prob).data.clone()

    def get_loss(volatile=False):
        action_prob = policy_net(Variable(states, volatile=volatile))
        log_prob = log_density(Variable(actions), action_prob)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean() + entropy_coef * policy_net.entropy(states)


    def get_kl():
        prob1 = policy_net(Variable(states))

        prob0 = Variable(prob1.data)
        #kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        kl = prob0 * (torch.log(prob0) - torch.log(prob1))
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, max_kl, damping)

class TRPO(object):
    def __init__(self, obs_dim, act_dim, normalizer):
        self.policy_net = StochasticPolicy(obs_dim, act_dim, hidden_dim=300, normalizer=normalizer)
        self.value_net = Value(obs_dim, hidden_dim=300, normalizer=normalizer)

        self.type = 'TRPO'

    def get_actor(self):
        return self.policy_net

    def to_train(self):
        self.policy_net.train()
        self.value_net.train()

    def to_eval(self):
        self.policy_net.eval()
        self.value_net.eval()

    def cpu(self):
        self.policy_net.cpu()
        self.value_net.cpu()

    def to(self, device):
        self.policy_net.to(device)
        self.value_net.to(device)

    def train(self, batch, entropy_coef=1e-3, gamma=0.995, tau=0.97, l2_reg=1e-3, max_kl=1e-2, damping=1e-1):
        self.cpu()
        update_params(batch, self.policy_net, self.value_net, gamma, tau, l2_reg, max_kl, damping,
                        entropy_coef)

