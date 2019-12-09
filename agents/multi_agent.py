import torch
from agents.TD3.TD3 import TD3
from agents.PPO.PPO import PPO
from agents.TRPO.TRPO import TRPO
from models.agent import StochasticPolicy, Policy
from utils import Transition
from copy import deepcopy
from utils import device

class MultiAgent(object):
    def __init__(self, 
                obs_dim, 
                act_dim, 
                normalizer, 
                gamma=None, 
                tau=None, 
                num_inter=9, 
                pg_type='ppo'):
        if pg_type == 'ppo':
            self.policies = [PPO(obs_dim, 1, normalizer, gamma, tau) 
                                for _ in range(num_inter)]
        elif pg_type == 'trpo':
            self.policies = [TRPO(obs_dim, 1, normalizer) for _ in range(num_inter)]
        elif pg_type == 'td3':
            self.policies = [TD3(obs_dim, 1, iters=1000, normalizer=normalizer)
                                for _ in range(num_inter)]
        else:
            raise NotImplementedError

        self.num_inter = num_inter
        self.obs_dim = obs_dim
        self.act_dim = 1
        self.normalizer = normalizer
        self.pg_type = pg_type

    def get_actor(self):
        return [policy.get_actor() for policy in self.policies]

    def save_policies(self, steps, args):
        ps = '_{}'.format(args.ps) if args.ps != '' else ''
        filename = '{}{}_{}'.format(args.pg_type, ps, steps)
        params = {}
        for idx in range(self.num_inter):
            params['model_state_dict{}'.format(idx)] = \
                self.policies[idx].get_actor().state_dict()
        torch.save(params, './model_log/' + filename)

    def load_policies(self, filename):
        checkpoint = torch.load('./model_log/' + filename)
        for idx in range(self.num_inter):
            self.policies[idx].get_actor().load_state_dict(checkpoint['model_state_dict{}'.format(idx)])

    def train(self, batches):
        for idx in range(self.num_inter):
            if self.pg_type == 'td3':
                self.policies[idx].buffer_add(batches[idx])
                self.policies[idx].train()
            else:
                self.policies[idx].train(batches[idx])
