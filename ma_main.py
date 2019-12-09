from env.flow_lib import flow_env
from utils import parser, log
from utils.normalizer import Normalizer
from agents.TD3.TD3 import TD3
from agents.PPO.PPO import PPO
from agents.TRPO.TRPO import TRPO
from agents.multi_agent import MultiAgent
from utils.rollout import real_batch, evaluate, ma_evaluate, ma_batch
from utils import Transition, device

import numpy as np
import gym
import gym.spaces

import torch
import torch.optim as optim
import torch.nn as nn

torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

NUM_INTER = 9
args = parser.parser()
print('agent type: {}'.format(args.pg_type))
env, env_name = flow_env(render=args.render, use_inflows=True, horizon=4000)

### seeding ###
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
###############

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
tb_writer, label = log.log_writer(args)
total_steps = 0
normalizer = Normalizer(obs_dim)
print("simulated task: {}".format(env_name))

policies = MultiAgent(obs_dim, act_dim, normalizer, args.gamma, args.tau, NUM_INTER,
                        args.pg_type)

for i_episode in range(args.num_episodes):
    policies.save_policies(total_steps, args)

    ### evaluation
    avg_reward = ma_evaluate(policies.get_actor(), env, batch_size=1000)
    tb_writer.add_scalar('{}/{}'.format(env_name, 'eval_mean'), avg_reward, total_steps)
    print('Episode: {}, Perf: {:.3f}'.format(i_episode + 1, avg_reward))

    ### sampling from environment    
    batches = ma_batch(policies.get_actor(), env, args.batch_size)
    total_steps += args.batch_size
    policies.train(batches)
