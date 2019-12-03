import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import numpy as np

from utils import device, MAX_LOOP
from utils import Transition

def rollout(policy, env, batch_size):
    ### the following three lists are used to update the dynamic network
    policy.to(device)
    states = []
    actions = []
    next_states = []
    masks = []
    rewards = []

    num_steps = 0
    reward_batch = 0
    num_episodes = 0

    while num_steps < batch_size:
        state = env.reset()
        #states.append(np.array([state]))
        states.append(state)
        reward = []

        for t in range(10000): # Don't infinite loop while learning
            s = torch.from_numpy(state.reshape(1, -1)).to(device)
            action = select_action(policy, s)
            action = action.cpu().data[0].numpy()
            action = np.clip(action, -1, 1)
            actions.append(action)
            next_state, r, done, _ = env.step(action)
            next_states.append(next_state)
            reward.append(r)

            mask = 1
            if done:
                mask = 0
            masks.append(mask)

            if done:
                break

            states.append(next_state)
            state = next_state
        num_steps += (t+1)
        num_episodes += 1
        rewards.append(reward)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    masks = np.array(masks)
    return states, actions, next_states, masks, rewards

def evaluate(policy, env, batch_size):
    _, _, _, _, rewards = rollout(policy, env, batch_size)
    rewards = [sum(item) for item in rewards]
    return np.mean(rewards), np.std(rewards) 

def real_batch(policy, env, batch_size):
    states, actions, next_states, masks, rewards = rollout(policy, env, batch_size)
    rewards = np.array([item for sublist in rewards for item in sublist])
    batch = Transition(states,
                        actions,
                        masks,
                        next_states,
                        rewards)
    return batch

def select_action(policy, state):
    if policy.type == 'stochastic':
        p = policy(state)
        m = Bernoulli(p)
        action = m.sample()
    else:
        p = policy(state)
        action = (p > 0.5).double()
        m = Bernoulli(torch.ones(p.shape) * 0.4)
        noise = m.sample().to(device)
        action = ((noise - action) != 0).double()

    action = torch.clamp(action, min=0, max=1)
    return action
