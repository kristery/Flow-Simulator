import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from models.agent import StochasticPolicy, Value
from agents.PPO.utils import log_density, gae
from utils import device


def surrogate_loss(policy_net, advantages, states, old_policy, actions):
    mean, log_std, std = policy_net(states)
    new_policy = log_density(actions, mean, std, log_std)
    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advantages
    return surrogate, ratio


class PPO(object):
    def __init__(self, obs_dim, act_dim, normalizer, gamma, tau):
        self.policy_net = StochasticPolicy(obs_dim, act_dim, 300, normalizer).to(device)
        self.value_net = Value(obs_dim, hidden_dim=300, normalizer=normalizer).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        self.type = 'PPO'
        self.gamma = gamma
        self.tau = tau

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

    def train(self, batch, entropy_coef=1e-3, n_iter=1, batch_size=16, clip_param=0.2):
        states = torch.Tensor(batch.state).to(device)
        actions = torch.Tensor(batch.action).to(device)
        returns, advantages = gae(batch, self.value_net, self.gamma, self.tau)

        #returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        mean, log_std, std = self.policy_net(states)
        old_policy = log_density(actions, mean, std, log_std).detach()
        old_values = self.value_net(states).detach()

        for _ in range(n_iter):
            index = np.random.permutation(returns.shape[0]) 
            index = np.array_split(index, returns.shape[0] // batch_size)
            for idx in index:
                batch_states = states[idx, :]
                batch_actions = actions[idx, :]
                batch_returns = returns[idx, :]
                batch_advantages = advantages[idx, :]
                batch_old_values = old_values[idx, :]
                batch_old_policy = old_policy[idx, :]

                loss, ratio = surrogate_loss(self.policy_net, batch_advantages, batch_states,
                                            batch_old_policy, batch_actions)

                values = self.value_net(batch_states)
                clipped_values = batch_old_values + \
                                torch.clamp(values - batch_old_values, -clip_param, clip_param)
                value_loss1 = (clipped_values - batch_returns).pow(2)
                value_loss2 = (values - batch_returns).pow(2)
                value_loss = torch.max(value_loss1, value_loss2).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
                clipped_loss = clipped_ratio * batch_advantages
                loss = -torch.min(loss, clipped_loss).mean()

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
