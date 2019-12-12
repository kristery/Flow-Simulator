from env.flow_lib import flow_env
import torch
from torch.distributions.bernoulli import Bernoulli
import numpy as np
from utils import device
from utils.rollout import ma_evaluate
from utils.normalizer import Normalizer
from models.agent import StochasticPolicy, Policy
from agents.multi_agent import MultiAgent

env, env_name = flow_env(render=True, use_inflows=True, sim_step=1, horizon=5000)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
print(obs_dim)
normalizer = Normalizer(obs_dim)

filename = 'ppo_4530000'
#filename = 'ppo_0'
#filename = 'td3_shortgreenpenalty_1332000'
### load RL policy ###
policies = MultiAgent(obs_dim, 1, normalizer, 0.995, 0.9) 
policies.load_policies(filename)
reward_sum = 0.

ma_evaluate(policies.get_actor(), env, 2000)
total_wait_time = 0.
for key in env.eval_time:
    total_wait_time += env.eval_time[key]
print('avg waiting time: {}'.format(total_wait_time / len(env.eval_time.keys())))
"""
for i in range(1):
    
    state = env.reset()
    for j in range(100000):
        s = torch.from_numpy(state.reshape(1, -1)).float().to(device)
        #print(actor(s))
        if 'ppo' in filename:
            m = Bernoulli(actor(s))
            a = torch.clamp(m.sample(), min=0, max=1)
            print(actor(s))
        elif 'td3' in filename:
            a = actor(s)
            a = (a > 0.5).float()
        action = a.cpu().data[0].numpy()
        #print('prob:', actor(s))
        next_state, reward, done, _ = env.step(action)
        #reward_sum += reward
        state = next_state
        if done:
            break
#print('total_reward: {}'.format(reward_sum))
"""
