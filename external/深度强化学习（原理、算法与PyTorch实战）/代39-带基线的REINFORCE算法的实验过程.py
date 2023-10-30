# 代39-带基线的REINFORCE算法的实验过程

import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

render = False
class Policy(nn.Module):
    def __init__(self,n_states, n_hidden, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)

 #这是policy的参数
        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []
#这是state_action_func的参数
        # self.Reward = []
        # self.s_value = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        output = F.softmax(self.linear2(x), dim= 1)
        # self.act_probs.append(action_probs)
        return output
env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
policy = Policy(n_states, 128, n_actions)
s_value_func = Policy(n_states, 128, 1)
alpha_theta = 1e-3
optimizer_theta = optim.Adam(policy.parameters(), lr=alpha_theta)
gamma = 0.99
seed = 1
env.seed(seed)
torch.manual_seed(seed)
live_time = []
def plot(epi, run_time):

    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Run Time')
    plt.plot(epi, run_time)
    plt.show()
if __name__ == '__main__':
    running_reward = 10
    i_episodes = []
    for i_episode in range(1, 10000):
        state, ep_reward = env.reset(), 0
        if render: env.render()
        policy_loss = []
        s_value = []
        state_sequence = []
        log_act_prob = []
        for t in range(10000):
            state = torch.from_numpy(state).unsqueeze(0).float()  # 在第0维增加一个维度，将数据组织成[N , .....] 形式
            state_sequence.append(deepcopy(state))
            action_probs = policy(state)
            m = Categorical(action_probs)
            action = m.sample()
            m_log_prob = m.log_prob(action)
            log_act_prob.append(m_log_prob)
            # policy.log_act_probs.append(m_log_prob)
            action = action.item()
            next_state, re, done, _ = env.step(action)
            if render: env.render()
            policy.reward.append(re)
            ep_reward += re
            if done:
                break
            state = next_state
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        i_episodes.append(i_episode)
        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:2f}\tAverage length: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        live_time.append(running_reward)
        R = 0
        Gt = []

        # get Gt value
        for r in policy.reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        # update step by step
        for i in range(len(Gt)):
            G = Gt[i]
            V = s_value_func(state_sequence[i])
            delta = G - V

            # update value network
            alpha_w = 1e-3  # 初始化

            optimizer_w = optim.Adam(s_value_func.parameters(), lr=alpha_w)
            optimizer_w.zero_grad()
            policy_loss_w = -delta
            policy_loss_w.backward(retain_graph=True)
            clip_grad_norm_(policy_loss_w, 0.1)
            optimizer_w.step()

            # update policy network
            optimizer_theta.zero_grad()
            policy_loss_theta = - log_act_prob[i] * delta
            policy_loss_theta.backward(retain_graph=True)
            clip_grad_norm_(policy_loss_theta, 0.1)
            optimizer_theta.step()
        del policy.log_act_probs[:]
        del policy.reward[:]
        if (i_episode % 1000 == 0):
            plot(i_episodes, live_time)
    np.save(f"withB", live_time)
    #policy.plot(live_time)