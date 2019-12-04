from env.flow_lib import flow_env
from torch.distributions.bernoulli import Bernoulli
import torch
import numpy as np

env, env_name = flow_env(render=False, use_inflows=True)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape
print(env.observation_space.shape)
rewards = []

ratios = [0.1 * float(item) for item in range(10)]

for ratio in ratios:
    state = env.reset()
    reward_sum = 0
    for j in range(100000):
        #action = np.zeros(act_dim)
        #action = np.random.choice(2, act_dim)
        m = Bernoulli(torch.ones(act_dim) * ratio)
        action = m.sample().detach().numpy()
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            break
    rewards.append(reward_sum)

for i in range(len(ratios)):
    print('{:.1f}\t{}\n'.format(ratios[i], rewards[i]))
