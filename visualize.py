from env.flow_lib import flow_env
import torch
from torch.distributions.bernoulli import Bernoulli
import numpy as np
from utils import device
from utils.normalizer import Normalizer
from models.agent import StochasticPolicy, Policy

env, env_name = flow_env(render=False, use_inflows=True)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
print(obs_dim)
normalizer = Normalizer(obs_dim)

filename = 'ppo_340000'
#filename = 'td3_shortgreenpenalty_1332000'
### load RL policy ###
if 'ppo' in filename:
    actor = StochasticPolicy(obs_dim, act_dim, 300, normalizer=normalizer).to(device)
elif 'td3' in filename:
    actor = Policy(obs_dim, act_dim, hidden_dim=400, normalizer=normalizer).to(device)
else:
    raise NotImplementedError

checkpoint = torch.load('./model_log/' + filename)
actor.load_state_dict(checkpoint['model_state_dict'])
reward_sum = 0.

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
        reward_sum += reward
        state = next_state
        if done:
            break
print('total_reward: {}'.format(reward_sum))

