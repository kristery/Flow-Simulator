from env.flow_lib import flow_env
import torch
import numpy as np
from utils import device
from utils.normalizer import Normalizer
from models.agent import StochasticPolicy, Policy

env, env_name = flow_env(render=False, use_inflows=True)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
normalizer = Normalizer(obs_dim)

filename = 'ppo_1880000'
### load RL policy ###
if 'ppo' in filename:
    actor = StochasticPolicy(obs_dim, act_dim, 300, normalizer=normalizer).to(device)
else:
    raise NotImplementedError

checkpoint = torch.load('./model_log/' + filename)
actor.load_state_dict(checkpoint['model_state_dict'])

for i in range(1):
    state = env.reset()
    for j in range(100000):
        s = torch.from_numpy(state.reshape(1, -1)).float().to(device)
        a = (actor(s) > 0.5).double()
        action = a.cpu().data[0].numpy()
        print(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

