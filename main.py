from env.flow_lib import flow_env
from utils import parser, log
from utils.normalizer import Normalizer
from agents.TD3.TD3 import TD3
from agents.PPO.PPO import PPO
from agents.TRPO.TRPO import TRPO
from utils.rollout import real_batch, evaluate
from utils import Transition, device

import numpy as np
import gym
import gym.spaces

import torch
import torch.optim as optim
import torch.nn as nn

torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

args = parser.parser()
print('agent type: {}'.format(args.pg_type))
env, env_name = flow_env(render=args.render, use_inflows=True)

### seeding ###
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
###############

def save_policy(steps, actor):
    ps = '_{}'.format(args.ps) if args.ps != '' else ''
    filename = '{}{}_{}'.format(args.pg_type, ps, steps)
    torch.save({
                'steps': steps,
                'model_state_dict': actor.state_dict()
                }, './model_log/'+filename)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
tb_writer, label = log.log_writer(args)
total_steps = 0
normalizer = Normalizer(obs_dim)
print("simulated task: {}".format(env_name))

if args.pg_type == 'ppo':
    policy = PPO(obs_dim, act_dim, normalizer, args.gamma, args.tau)
elif args.pg_type == 'trpo':
    policy = TRPO(obs_dim, act_dim, normalizer=normalizer)
elif args.pg_type == 'td3':
    policy = TD3(obs_dim, act_dim, iters=1000, normalizer=normalizer)
else:
    assert NotImplementedError

for i_episode in range(args.num_episodes):
    save_policy(total_steps, policy.get_actor())

    ### evaluation
    reward_mean, reward_std = evaluate(policy.get_actor(), env, batch_size=1000)
    tb_writer.add_scalar('{}/{}'.format(env_name, 'eval_mean'), reward_mean, total_steps)
    tb_writer.add_scalar('{}/{}'.format(env_name, 'eval_std'), reward_std, total_steps)
    print('Episode: {}, Perf: {:.3f}, Std: {:.3f}'.format(i_episode + 1, reward_mean, reward_std))

    ### sampling from environment    
    batch = real_batch(policy.get_actor(), env, args.batch_size)
    real_states = torch.Tensor(batch.state).to(device)
    total_steps += real_states.shape[0]
    
    if args.pg_type == 'td3':
        policy.buffer_add(batch)
        policy.train()
    else:
        policy.train(batch)
