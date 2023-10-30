'''深度强化学习——原理、算法与PyTorch实战，代码名称：代42-A3C算法的实验结果及分析.py'''

# 导入相应的模块
import random
import gym
import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.distributions import Normal
    
# ActorCritic网络定义
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, device):
        '''构造函数，定义ActorCritic网络中各layer'''
        super().__init__()
        self.act_limit = torch.as_tensor(act_limit, dtype=torch.float32, device=device)
        self.value_layer1 = nn.Linear(obs_dim, 256)
        self.value_layer2 = nn.Linear(256, 1)
        self.policy_layer1 = nn.Linear(obs_dim, 256)
        self.mu_layer = nn.Linear(256, act_dim)
        self.sigma_layer = nn.Linear(256, act_dim)

    def forward(self, obs):
        '''前馈函数，定义ActorCritic网络向前传播的方式'''
        value = F.relu6(self.value_layer1(obs))
        value = self.value_layer2(value)
        policy = F.relu6(self.policy_layer1(obs))
        mu = torch.tanh(self.mu_layer(policy)) * self.act_limit
        sigma = F.softplus(self.sigma_layer(policy))
        return value, mu, sigma

    def select_action(self, obs):
        '''采用随机高斯策略选择动作'''
        _, mu, sigma = self.forward(obs)
        pi = Normal(mu, sigma)
        return pi.sample().cpu().numpy()

    def loss_func(self, states, actions, v_t, beta):
        '''计算损失函数'''
        # 计算值损失
        values, mu, sigma = self.forward(states)
        td = v_t - values
        value_loss = torch.squeeze(td ** 2)
        # 计算熵损失
        pi = Normal(mu, sigma)
        log_prob = pi.log_prob(actions).sum(axis=-1)
        entropy = pi.entropy().sum(axis=-1)
        policy_loss = -(log_prob * torch.squeeze(td.detach()) + beta * entropy)
        return (value_loss + policy_loss).mean()


# Worker
class Worker(mp.Process):
    def __init__(self, id, device, env_name, obs_dim, act_dim, act_limit,
                 global_network, global_optimizer,
                 gamma, beta, global_T, global_T_MAX, t_MAX,
                 global_episode, global_return_display, global_return_record, global_return_display_record):
        '''构造函数，定义Worker中各参数'''
        super().__init__()
        self.id = id
        self.device = device
        self.env = gym.make(env_name)
        # self.env.seed(seed)
        self.local_network = ActorCritic(obs_dim, act_dim, act_limit, self.device).to(self.device)
        self.global_network = global_network
        self.global_optimizer = global_optimizer
        self.gamma, self.beta = gamma, beta
        self.global_T, self.global_T_MAX, self.t_MAX = global_T, global_T_MAX, t_MAX
        self.global_episode = global_episode
        self.global_return_display = global_return_display
        self.global_return_record = global_return_record
        self.global_return_display_record = global_return_display_record

    def update_global(self, states, actions, rewards, next_states, done, gamma, beta, optimizer):
        '''更新一次全局梯度信息'''
        if done:
            R = 0
        else:
            R, mu, sigma = self.local_network.forward(next_states[-1])
        length = rewards.size()[0]

        # 计算真实值
        v_t = torch.zeros([length, 1], dtype=torch.float32, device=self.device)
        for i in range(length, 0, -1):
            R = rewards[i - 1] + gamma * R
            v_t[i - 1] = R

        loss = self.local_network.loss_func(states, actions, v_t, beta)  #计算损失函数
        # 使用异步并行的工作组进行梯度累积，对全局网络进行异步更新
        optimizer.zero_grad()
        loss.backward()
        for local_params, global_params in zip(self.local_network.parameters(), self.global_network.parameters()):
            global_params._grad = local_params._grad
        optimizer.step()
        self.local_network.load_state_dict(self.global_network.state_dict())

    def run(self):
        '''运行游戏情节'''
        # 初始化环境和情节奖赏
        t = 0
        state, done = self.env.reset(), False
        episode_return = 0

        # 获取一个buffer的数据
        while self.global_T.value <= self.global_T_MAX:
            t_start = t
            buffer_states, buffer_actions, buffer_rewards, buffer_next_states = [], [], [], []
            while not done and t - t_start != self.t_MAX:
                action = self.local_network.select_action(torch.as_tensor(state, dtype=torch.float32, device=self.device))
                next_state, reward, done, _ = self.env.step(action)
                episode_return += reward
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_next_states.append(next_state)
                buffer_rewards.append(reward / 10)
                t += 1
                with self.global_T.get_lock():
                    self.global_T.value += 1
                state = next_state

            # 根据这一buffer的数据来更新全局梯度信息
            self.update_global(
                torch.as_tensor(buffer_states, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_actions, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_rewards, dtype=torch.float32, device=self.device),
                torch.as_tensor(buffer_next_states, dtype=torch.float32, device=self.device),
                done, self.gamma, self.beta, self.global_optimizer
            )

            # 处理情节完成时的操作
            if done:
                # global_episode上锁，处理情节完成时的操作
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
                    self.global_return_record.append(episode_return)
                    if self.global_episode.value == 1:
                        self.global_return_display.value = episode_return  # 第一次完成情节
                    else:
                        self.global_return_display.value *= 0.99
                        self.global_return_display.value += 0.01 * episode_return
                        self.global_return_display_record.append(self.global_return_display.value)
                        # 10个情节输出一次
                        if self.global_episode.value % 10 == 0:
                            print('Process: ', self.id, '\tepisode: ', self.global_episode.value, '\tepisode_return: ', self.global_return_display.value)
                episode_return = 0
                state, done = self.env.reset(), False

if __name__ == "__main__":

    # 定义实验参数
    device = 'cuda:1'
    env_name = 'Pendulum-v0'
    num_processes = 8
    gamma = 0.9
    beta = 0.01
    lr = 1e-4
    T_MAX = 1000000
    t_MAX = 5

    # 设置进程启动方式为spawn
    mp.set_start_method('spawn')

    # 定义环境和进程相关参数
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high
    global_network = ActorCritic(obs_dim, act_dim, act_limit, device).to(device)
    global_network.share_memory()
    optimizer = Adam(global_network.parameters(), lr=lr)
    global_episode = mp.Value('i', 0)
    global_T = mp.Value('i', 0)
    global_return_display = mp.Value('d', 0)
    global_return_record = mp.Manager().list()
    global_return_display_record = mp.Manager().list()

    # 定义worker 各worker进程开始训练
    workers = [Worker(i, device, env_name, obs_dim, act_dim, act_limit, \
                      global_network, optimizer, gamma, beta,
                      global_T, T_MAX, t_MAX, global_episode,
                      global_return_display, global_return_record, global_return_display_record) for i in range(num_processes)]

    [worker.start() for worker in workers]
    [worker.join() for worker in workers]

    # 保存模型
    torch.save(global_network, 'a3c_model.pth')
    
    # 实验结果可视化
    import matplotlib.pyplot as plt
    save_name = 'a3c_gamma=' + str(gamma) + '_beta=' + str(beta) + '_' + env_name
    save_data = np.array(global_return_record)
    plt.plot(np.array(global_return_display_record))
    np.save(save_name + '.npy', save_data)
    plt.ylabel('return')
    plt.xlabel('episode')
    plt.savefig(save_name + '.png')