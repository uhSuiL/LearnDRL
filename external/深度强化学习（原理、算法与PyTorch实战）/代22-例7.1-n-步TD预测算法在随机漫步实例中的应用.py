''' 深度强化学习——原理、算法与PyTorch实战，代码名称：代22-例7.1-n-步TD预测算法在随机漫步实例中的应用.py '''

# 导入相应的模块
from random import random, choice
from myBookEnvCode.my_random_walk import RandomWalkEnv
import matplotlib.pyplot as plt
import math
import numpy as np
import max_min_smooth_py

nums_for_ave = 1000  #平均1000次


class Agent():
    def __init__(self):
        self.actions = [0, 1]  # 两个动作
        self.aveErrors = {}
        ''' 不同n时的平均均方误差 '''
        self.aveErrors[1] = np.zeros(100)
        self.aveErrors[2] = np.zeros(100)
        self.aveErrors[4] = np.zeros(100)
        self.aveErrors[8] = np.zeros(100)
        self.epErrors = []

    def calc_error(self, ep_num, V):
        ''' 计算平均均方误差 '''
        error = 0
        for i in range(1, 10):
            target = -1 + 0.2 * i
            data = V[i] if i in V.keys() else 0
            error = error + (target - data) * (target - data)
        error = math.sqrt(error / 9)
        self.epErrors[ep_num] = error

    def nStepTDLearning(self, env, n, gamma, alpha, max_episode_num):
        ''' n步Td算法学习 '''
        self.V = {}
        self.epErrors = np.zeros(max_episode_num)
        num_episode = 1
        ''' 运行max_episode_num情节 '''
        while num_episode <= max_episode_num:
            s = env.reset()  # 环境初始化
            if not s in self.V.keys():
                self.V[s] = 0
            a = self.Policy(s, num_episode)  # 选择动作
            time_in_episode = 0
            T_ep = 1000
            ep_states = [s]
            ep_actions = [a]
            ep_rewards = [0]  #奖赏下标从1开始
            time_update = 0
            while True:
                if time_in_episode < T_ep:
                    n_s, n_r, done, _ = env.step(a)  # 进入下一个状态
                    env.render()
                    if not n_s in self.V.keys():  # 没有访问过的状态，值初始化为0
                        self.V[n_s] = 0
                    ep_states.append(n_s)
                    ep_rewards.append(n_r)
                    if done:
                        T_ep = time_in_episode + 1
                    else:
                        n_a = self.Policy(n_s, num_episode)
                        ep_actions.append(n_a)
                        s = n_s
                        a = n_a
                time_update = time_in_episode - n + 1

                ''' 判断，进行更新 '''
                if time_update >= 0:
                    n_step_return = 0
                    xs = 1
                    for i in range(time_update + 1,
                                   min(time_update + n + 1, T_ep + 1)):
                        n_step_return += xs * ep_rewards[i] 
                        xs *= gamma
                    if (time_update + n < T_ep):
                        n_step_return += xs * self.__getVValue(
                            ep_states[time_update + n])
                    ''' 获取旧的v值，更新后写进v表 '''
                    old_v = self.__getVValue(ep_states[time_update])
                    new_v = old_v + alpha * (n_step_return - old_v)
                    self.__setVValue(ep_states[time_update], new_v)
                time_in_episode += 1
                if (time_update == T_ep - 1):
                    break

            self.calc_error(num_episode - 1, self.V)  # 每个情节结束计算一次误差
            num_episode += 1
        self.aveErrors[n][int(alpha * 100) - 1] += sum(
            self.epErrors) / max_episode_num  # 所有情节误差进行平均
        return

    def Policy(self, s, episode_num):
        ''' 定义策略 '''
        a = choice(list(self.actions))  # 左右走概率均等的策略
        return a

    def __getVValue(self, s):  # ①
        ''' V值的获取 '''
        return self.V[s]  # argmax(q)

    def __setVValue(self, s, new_v):  # ②
        ''' V值的设置 '''
        self.V[s] = new_v


def dealData(aveErrors):
    ''' 处理数据 '''
    aveErrors = max_min_smooth_py.max_min_smooth(aveErrors)
    for i in range(len(aveErrors)):
        aveErrors[i] = aveErrors[i] / nums_for_ave
    return aveErrors


def plot_n_alpha(aveErrors):
    ''' 绘制图像 '''
    x_axis = np.linspace(0.01, 1, 100)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 用bottom代替x轴
    ax.yaxis.set_ticks_position('left')  # 用left代替y轴
    
    '''不同n的取值时的平均均方根误差'''
    plt.plot(x_axis,
             dealData(aveErrors[1]),
             color="green",
             ls='-',
             label="$n=1$")
    plt.plot(x_axis,
             dealData(aveErrors[2]),
             color="red",
             ls='--',
             label="$n=2$")
    plt.plot(x_axis,
             dealData(aveErrors[4]),
             color="skyblue",
             ls='-.',
             label="$n=4$")
    plt.plot(x_axis,
             dealData(aveErrors[8]),
             color="blue",
             ls=':',
             label="$n=8$")

    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("平均均方根误差")
    plt.show()


def nStepTDLearningExample(agent, env):
    ''' 取不同学习率进行实验 '''
    x = np.linspace(0.01, 1, 100)
    for alpha_num in x:  # 不同n值的实验结果
        for _ in range(nums_for_ave):
            agent.nStepTDLearning(env=env,
                                  n=1,
                                  gamma=1,
                                  alpha=alpha_num,
                                  max_episode_num=6)
        for _ in range(nums_for_ave):
            agent.nStepTDLearning(env=env,
                                  n=2,
                                  gamma=1,
                                  alpha=alpha_num,
                                  max_episode_num=6)
        for _ in range(nums_for_ave):
            agent.nStepTDLearning(env=env,
                                  n=4,
                                  gamma=1,
                                  alpha=alpha_num,
                                  max_episode_num=6)
        for _ in range(nums_for_ave):
            agent.nStepTDLearning(env=env,
                                  n=8,
                                  gamma=1,
                                  alpha=alpha_num,
                                  max_episode_num=6)
    #根据不同的步数n以及学习率alpha绘制图像
    plot_n_alpha(agent.aveErrors)


if __name__ == "__main__":
    agent = Agent()
    env = RandomWalkEnv()  # 引入环境env
    print("Learning...")
    nStepTDLearningExample(agent, env)
