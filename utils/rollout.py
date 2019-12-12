import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import numpy as np

from utils import device, MAX_LOOP
from utils import Transition

def ma_rollout(policies, env, batch_size):
    num_inter = len(policies)
    for policy in policies:
        policy.to(device)

    count_flag = np.zeros(num_inter)
    count_down = np.zeros(num_inter)
    temp_rewards = [[] for _ in range(num_inter)]
    states = [[] for _ in range(num_inter)]
    actions = [[] for _ in range(num_inter)]
    next_states = [[] for _ in range(num_inter)]
    masks = [[] for _ in range(num_inter)]
    rewards = [[] for _ in range(num_inter)]

    num_steps = 0
    reward_batch = []
    num_episodes = 0
    reward_sum = 0.

    while num_steps < batch_size:
        info = [1. for _ in range(num_inter)]
        state = env.reset()
        #states.append(state)
        for t in range(10000): # Don't infinite loop while learning
            s = torch.from_numpy(state).to(device)
            a = []
            for idx in range(len(info)):
                if info[idx] == 1:
                    action = select_action(policies[idx], s[idx].unsqueeze(0))
                    count_down[idx] = action
                    count_flag[idx] = 1
                    #print(state[idx], [action])
                    states[idx].append(state[idx])
                    actions[idx].append(action)
                    masks[idx].append(1.)

                if count_flag[idx] == 1:
                    if count_down[idx] - env.sim_step < 0:
                        a.append(1.)
                        count_flag[idx] = 0
                        count_down[idx] = 0
                    else:
                        count_down[idx] -= env.sim_step
                        a.append(0.)
                else:
                    a.append(0.)
            next_state, r, done, info = env.step(np.array(a))
            reward_sum += sum(r)
            for idx in range(len(info)):
                temp_rewards[idx].append(r[idx])
                if info[idx] == 1:
                    next_states[idx].append(next_state[idx])
                    rewards[idx].append(np.mean(temp_rewards[idx]))
                    temp_rewards[idx] = []
            #print([len(item) for item in states], 
            #        [len(item) for item in next_states], done)
            if done:
                for idx in range(len(masks)):
                    masks[idx][-1] = 0.
                    if len(states[idx]) == (len(next_states[idx]) + 1):
                        next_states[idx].append(next_state[idx])
                        rewards[idx].append(np.mean(temp_rewards[idx]))
                        temp_rewards[idx] = []
                    elif len(states[idx]) != len(next_states[idx]):
                        print(len(states[idx]), len(next_states[idx]), idx)
                        raise Exception
                break
        num_steps += (t+1)
        num_episodes += 1
        
    return states, actions, next_states, masks, rewards, reward_sum / num_episodes

def ma_evaluate(policies, env, batch_size):
    _, _, _, _, _, avg_rewards = ma_rollout(policies, env, batch_size)
    return avg_rewards

def ma_batch(policies, env, batch_size):
    states, actions, next_states, masks, rewards, avg_reward = ma_rollout(policies, 
                                                                        env, batch_size)
    batches = []
    for idx in range(len(states)):
        batches.append(Transition(np.array(states[idx]),
                                    np.array(actions[idx]),
                                    np.array(masks[idx]).reshape(-1),
                                    np.array(next_states[idx]),
                                    np.array(rewards[idx]).reshape(-1)))
    return batches


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
        mean, _, std = policy(state)
        action = torch.normal(mean, std)
    else:
        mean = policy(state)
        std = torch.normal(torch.ones(mean.shape) * 0.1).to(device)
        action = mean + std

    action = torch.clamp(action, min=-1, max=1)
    # length of green light
    # 15 ~ 125
    action = (action * 55.) + 70
    return action.cpu().data[0].numpy()
