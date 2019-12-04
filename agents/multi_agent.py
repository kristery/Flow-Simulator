import torch
from agents.TD3.TD3 import TD3
from agents.PPO.PPO import PPO
from agents.TRPO.TRPO import TRPO
from models.agent import StochasticPolicy, Policy
from utils import Transition
from copy import deepcopy

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
            self.policies = [PPO(obs_dim, act_dim, normalizer, gamma, tau) 
                                for _ in range(num_inter)]
        elif pg_type == 'trpo':
            self.policies = [TRPO(obs_dim, act_dim, normalizer) for _ in range(num_inter)]
        elif pg_type == 'td3':
            self.policies = [TD3(obs_dim, act_dim, iters=1000, normalizer=normalizer)
                                for _ in range(num_inter)]
        else:
            raise NotImplementedError

        self.num_inter = num_inter
        self.obs_dim = obs_dim // num_inter
        self.act_dim = 1
        self.normalizer = normalizer
        self.pg_type = pg_type

    def save_policies(self, steps, args):
        ps = '_{}'.format(args.ps) if args.ps != '' else ''
        filename = '{}{}_{}'.format(args.pg_type, ps, steps)
        params = {}
        for idx in range(self.num_inter):
            params['model_state_dict{}'.format(idx)] = \
                self.policies[idx].get_actor().state_dict()
        torch.save(params, './model_log/' + filename)

    def load_policies(self, filename):
        self.policies = [StochasticPolicy(self.obs_dim, self.act_dim, 300, 
                            self.normalizer).to(device) for _ in range(self.num_inter)]
        checkpoint = torch.load('./model_log/' + filename)
        for idx in range(self.num_inter):
            self.policies[idx].load_state_dict(checkpoint['model_state_dict{}'.format(idx)])

    def _batch_split(self, batch):
        num_observed = 4
        size_feature1 = num_observed * 4 * 4
        size_feature2 = 1
        for i in range(self.num_inter):
            state = np.concatenate([batch.state[:, 
                size_feature1 * i + self.num_inter * size_feature1 * j:
                size_feature1 * (i+1) + self.num_inter * size_featuere1 * j] 
                for j in range(self.num_inter)] + 
                [batch.state[:, j::self.num_inter] for j in range(self.num_inter)], axis=1)
            action = batch.action[:, i]
        

    def train()
