from env.flow_lib import flow_env
import numpy as np

env, env_name = flow_env(render=True, use_inflows=True)
print("simulated task: {}".format(env_name))

act_dim = env.action_space.shape
rewards0 = []
rewards1 = []

for i in range(1):
    state = env.reset()
    reward_sum = 0
    for j in range(100000):
        action = np.zeros(act_dim)
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        print(reward)
        if done:
            break
    rewards0.append(reward_sum)
"""
for i in range(3):
    state = env.reset()
    reward_sum = 0
    for j in range(100000):
        action = np.ones(act_dim)
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            break
    rewards1.append(reward_sum)
"""
#print('(All one baseline) Perf: {:.3f}, Std: {:.3f}'.format(np.mean(rewards1), np.std(rewards1)))
print('(All zero baseline) Perf: {:.3f}, Std: {:.3f}'.format(np.mean(rewards0), np.std(rewards0)))
