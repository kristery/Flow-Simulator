from env.flow_lib import flow_env
from torch.distributions.bernoulli import Bernoulli
import torch
import numpy as np

env, env_name = flow_env(render=False, use_inflows=True)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape
print(env.observation_space.shape)
rewards = []

ratios = [0.1, 0.1, 0.1]

for ratio in ratios:
    state = env.reset()
    for j in range(100000):
        #action = np.zeros(act_dim)
        #action = np.random.choice(2, act_dim)
        m = Bernoulli(torch.ones(act_dim) * ratio)
        action = m.sample().detach().numpy()
        next_state, reward, done, info = env.step(action)
        print(reward)
        print(info)
        #print(next_state)
        if done:
            break
