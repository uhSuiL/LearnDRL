''' 深度强化学习——原理、算法与PyTorch实战，代码名称：代24-例7.3-n-步TD预测、TD(0)以及TD(λ)算法应用于确定环境扫地机器人问题中的对比分析.py '''

# 导入相应的模块
from myBookEnvCode.my_book_gridworld0406 import GridWorldEnv
import numpy as np
from random import random, choice
import pandas as pd
from IPython.display import display


class nStepTDAgent():
    def __init__(self):
        self.V = {}  # 初始化V
        self.actions = [0, 1, 2, 3]  # 四个动作
        self.printTable = pd.DataFrame(
            columns=['S' + str(i) for i in range(25)])

    def saveEpisode(self, V, num_episode):
        ''' 展示当前情节结束状态值信息 '''
        v = []
        for i in range(25):
            v.append(format(V[i], '.3f') if i in V.keys() else 0)
        self.printTable.loc['V' + str(num_episode)] = v

    def nStepTDLearning(self, env, n, gamma, alpha, max_episode_num):
        ''' N步Td算法学习 '''
        num_episode = 0
        ''' 运行max_episode_num情节 '''
        while num_episode < max_episode_num:
            s = env.reset()  # 环境初始化
            if not s in self.V.keys():
                self.V[s] = 0  #初始化状态值
            a = self.Policy(s, env.valid_actions_for_states,
                            num_episode)  # 根据策略选择动作
            time_in_episode = 0
            T_ep = 1000
            ep_states = [s]
            ep_actions = [a]
            ep_rewards = [0]  #奖赏下标从1开始
            time_update = 0
            while True:
                if time_in_episode < T_ep:
                    n_s, n_r, done, _ = env.step(a)  # 进入一下个状态
                    if not n_s in self.V.keys():  # 没有访问过的状态，V值初始化为0
                        self.V[n_s] = 0
                    ep_states.append(n_s)
                    ep_rewards.append(n_r)
                    if done:
                        T_ep = time_in_episode + 1
                    else:
                        n_a = self.Policy(n_s, env.valid_actions_for_states,
                                          num_episode)
                        ep_actions.append(n_a)
                        s = n_s
                        a = n_a
                time_update = time_in_episode - n + 1
                #开始更新状态值
                if time_update >= 0:
                    n_step_return = 0
                    xs = 1
                    #计算n步回报
                    for i in range(time_update + 1,
                                   min(time_update + n + 1, T_ep + 1)):
                        n_step_return += xs * ep_rewards[i]
                        xs *= gamma
                    #未到达终止状态，进行自举
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

            num_episode += 1
            ''' 展示给定的情节结束状态值信息 '''
            self.saveEpisode(self.V, num_episode)
        pd.set_option('display.max_columns', 50)
        display(self.printTable)  # 打印表格
        return

    def Policy(self, s, action_list, episode_num):
        ''' 定义策略 '''
        if s == 24:
            a = 1
        else:
            a = 3
        return a

    def __getVValue(self, s):
        ''' V值的获取 '''
        return self.V[s]

    def __setVValue(self, s, new_v):
        ''' V值的设置 '''
        self.V[s] = new_v

    def __getEValue(self, s):
        ''' E值的获取 '''
        return self.E[s]

    def __setEValue(self, s, new_v):
        ''' E值的设置 '''
        self.E[s] = new_v


class nStepTDLambdaAgent():
    def __init__(self):
        self.V = {}  # 初始化V
        self.actions = [0, 1, 2, 3]  # 四个动作
        self.printTable = pd.DataFrame(
            columns=['S' + str(i) for i in range(25)])

    def saveEpisode(self, V, num_episode):
        ''' 展示当前情节结束状态值信息 '''
        v = []
        for i in range(25):
            v.append(V[i] if i in V.keys() else 0)
        self.printTable.loc['V' + str(num_episode)] = v

    def nStepTDLambdaLearning(self, env, lambda_, gamma, alpha,
                              max_episode_num):
        self.V = {}
        self.epErrors = np.zeros(max_episode_num)
        num_episode = 0
        while num_episode <= max_episode_num:
            self.E = {}
            s = env.reset()
            if not s in self.V.keys():
                self.V[s] = 0
            if not s in self.E.keys():
                self.E[s] = 0
            a = self.Policy(s, env.valid_actions_for_states, num_episode)
            time_in_episode = 0
            while True:
                n_s, n_r, done, _ = env.step(a)  # 进入一下个状态
                if not n_s in self.V.keys():
                    self.V[n_s] = 0
                if not n_s in self.E.keys():
                    self.E[n_s] = 0
                n_a = self.Policy(n_s, env.valid_actions_for_states,
                                  num_episode)
                v = self.__getVValue(s)  # 取V(s)
                v_next = self.__getVValue(n_s)
                delta = n_r + gamma * v_next - v

                e = self.__getEValue(s)  # 得到E值
                e = e + 1  # 每经历一个s，就为这个s下的Es增加1
                self.__setEValue(
                    s, e)  # 存下当前s的E。因为走一步就会有一个E，所以要先把该s存入，后面的遍历才可以访问到该s

                # TD(λ)，遍历所有经历过的s
                # 回溯之前的V。不讲究顺序，因为在一次游戏中，每前进一次，所有存在过的s都会被λ处理一次，出现的越早的s就会被alpha*e_value乘的越多，有值状态的更新就会表的越微小
                for s in self.E.keys():  # 遍历所有state
                    old_v = self.__getVValue(s)
                    e_value = self.__getEValue(s)
                    new_v = old_v + alpha * delta * e_value
                    new_e = gamma * lambda_ * e_value
                    self.__setVValue(s, new_v)  # 更新历史s的V
                    self.__setEValue(s, new_e)  # 更新

                s = n_s
                a = n_a
                time_in_episode += 1
                if done:
                    break

            num_episode += 1
            ''' 展示给定的情节结束状态值信息 '''
            self.saveEpisode(self.V, num_episode)
        pd.set_option('display.max_columns', 50)
        display(self.printTable)  # 打印表格
        return

    def Policy(self, s, action_list, episode_num):
        ''' 定义策略 '''
        if s == 24:
            a = 1
        else:
            a = 3
        return a

    def __getVValue(self, s):
        ''' V值的获取 '''
        return self.V[s]

    def __setVValue(self, s, new_v):
        ''' V值的设置 '''
        self.V[s] = new_v

    def __getEValue(self, s):
        ''' E值的获取 '''
        return self.E[s]

    def __setEValue(self, s, new_v):
        ''' E值的设置 '''
        self.E[s] = new_v


def nStepTDLearningExample(agent, env):
    agent.nStepTDLearning(env=env,
                          n=3,
                          gamma=0.8,
                          alpha=0.2,
                          max_episode_num=60)


def nStepTDLambdaLearningExample(agent, env):
    agent.nStepTDLambdaLearning(env=env,
                                lambda_=0.8,
                                gamma=0.8,
                                alpha=0.2,
                                max_episode_num=60)


if __name__ == "__main__":
    np.random.seed(1)
    nStepTDagent = nStepTDAgent()
    nStepTDLambdaagent = nStepTDLambdaAgent()
    env = GridWorldEnv()  # 引入环境env
    print("nStepTDLearning Learning...")
    nStepTDLearningExample(nStepTDagent, env)
    print("nStepTDLambdaagent Learning...")
    nStepTDLambdaLearningExample(nStepTDLambdaagent, env)

    