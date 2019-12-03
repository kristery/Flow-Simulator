from env.flow_lib import flow_env
from utils.rollout import evaluate
import torch
from torch.distributions.bernoulli import Bernoulli
import numpy as np
from utils import device
from utils.normalizer import Normalizer
from models.agent import StochasticPolicy, Policy

env, env_name = flow_env(render=False, use_inflows=True)
print("simulated task: {}".format(env_name))

env.seed(8021)
torch.manual_seed(8021)
np.random.seed(8021)

act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
normalizer = Normalizer(obs_dim)

filename = 'ppo_test_noent_88000'
### load RL policy ###
if 'ppo' in filename:
    actor = StochasticPolicy(obs_dim, act_dim, 300, normalizer=normalizer).to(device)
else:
    raise NotImplementedError

checkpoint = torch.load('./model_log/' + filename)
actor.load_state_dict(checkpoint['model_state_dict'])
actor.eval()
print(evaluate(actor.double(), env, 1000))
"""
reward_sum = 0.

for i in range(1):
    state = env.reset()
    for j in range(100000):
        s = torch.from_numpy(state.reshape(1, -1)).float().to(device)
        m = Bernoulli(actor(s))
        a = torch.clamp(m.sample(), min=0, max=1)
        action = a.cpu().data[0].numpy()
        #print('prob:', actor(s))
        next_state, reward, done, _ = env.step(action)
        print(action, reward)
        reward_sum += reward
        state = next_state
        if done:
            break
print('total_reward: {}'.format(reward_sum))
"""
