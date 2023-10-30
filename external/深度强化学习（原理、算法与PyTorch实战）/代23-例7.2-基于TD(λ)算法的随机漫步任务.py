''' 深度强化学习——原理、算法与PyTorch实战，代码名称：代23-例7.2-基于TD(λ)算法的随机漫步任务.py '''

# 导入相应的模块
from random import random, choice
from myBookEnvCode.my_random_walk import RandomWalkEnv
import math
import numpy as np
import max_min_smooth_py

lambda_dict = {0: 0, 0.4: 1, 0.8: 2, 0.9: 3}

f_alpha = lambda alpha: int(alpha * 100) - 1

nums_for_ave = 1000


class Agent():
    def __init__(self):
        self.actions = [0, 1]  # 两个动作
        self.aveErrors = {}
        ''' 不同lambda时的平均均方误差 '''
        for lambda_ in lambda_dict.keys():
            self.aveErrors[lambda_dict[lambda_]] = np.zeros(100)
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

    def nStepTDLambdaLearning(self, env, lambda_, gamma, alpha,
                              max_episode_num):
        ''' Td(lambda)算法学习 '''
        self.V = {}
        self.epErrors = np.zeros(max_episode_num)
        num_episode = 1
        ''' 运行max_episode_num情节 '''
        while num_episode <= max_episode_num:
            self.E = {}
            s = env.reset()  # 环境初始化
            if not s in self.V.keys():
                self.V[s] = 0  #初始化状态值
            if not s in self.E.keys():
                self.E[s] = 0  #初始化资格迹值
            a = self.Policy(s, num_episode)  # 根据策略选择动作
            time_in_episode = 0
            while True:
                n_s, n_r, done, _ = env.step(a)  # 进入下一个状态
                if not n_s in self.V.keys():  # 没有访问过的状态，V和E值初始化为0
                    self.V[n_s] = 0
                if not n_s in self.E.keys():
                    self.E[n_s] = 0
                n_a = self.Policy(n_s, num_episode)
                v = self.__getVValue(s)  # 取V(s)
                v_next = self.__getVValue(n_s)
                delta = n_r + gamma * v_next - v

                e = self.__getEValue(s)  # 得到E值
                e = e + 1  # 每经历一个s，就为这个s下的Es增加1
                self.__setEValue(
                    s, e)  # 存下当前s的E。因为走一步就会有一个E，所以要先把该s存入，后面的遍历才可以访问到该s

                # TD($\lambda$)，遍历所有经历过的s
                # 回溯之前的V。不讲究顺序，因为在一次游戏中，每前进一次，所有存在过的s都会被$\lambda$处理一次，出现的越早的s就会被alpha*e_value乘的越多，有值状态的更新就会表的越微小
                for s in self.E.keys():  # 遍历所有state
                    ''' 获取旧的v值，更新后写进v表 '''
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

            self.calc_error(num_episode - 1, self.V)  #每个情节结束计算一次误差
            num_episode += 1
        self.aveErrors[lambda_dict[lambda_]][f_alpha(alpha)] += sum(
            self.epErrors) / max_episode_num  #情节误差平均
        return

    # 左右走概率均等
    def Policy(self, s, episode_num):
        a = choice(list(self.actions))  # 随机选择动作
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


def dealData(aveErrors):
    ''' 数据输出处理，平滑化 '''
    aveErrors = max_min_smooth_py.max_min_smooth(aveErrors)
    for i in range(len(aveErrors)):
        aveErrors[i] = aveErrors[i] / nums_for_ave
        aveErrors[i] = min(aveErrors[i], 0.8)
    for k in range(3):
        max_data = max(aveErrors)
        for i in range(len(aveErrors)):
            if i > 1 and aveErrors[i] == max_data:
                aveErrors[i] = aveErrors[i - 1]
                break
    return aveErrors


def dealxy(x_axis, y):
    i = len(x_axis)
    while y[i - 1] == 0.8:
        i -= 1
    return x_axis[:i], y[:i]


def plot_lambda_alpha(aveErrors):
    ''' 绘制图像 '''
    x_axis = np.linspace(0.01, 1, 100)
    plt.ylim(0.2, 0.8)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 用bottom代替x轴
    ax.yaxis.set_ticks_position('left')  # 用left代替y轴
    '''不同lambda的取值时的平均均方根误差'''
    x_axis_new, aveData_new = dealxy(x_axis, dealData(aveErrors[0]))
    plt.plot(x_axis_new,
             aveData_new,
             color="green",
             ls='-',
             label=r"$\lambda=0$")
    x_axis_new, aveData_new = dealxy(x_axis, dealData(aveErrors[1]))
    plt.plot(x_axis_new,
             aveData_new,
             color="red",
             ls='--',
             label=r"$\lambda=0.4$")
    x_axis_new, aveData_new = dealxy(x_axis, dealData(aveErrors[2]))
    plt.plot(x_axis_new,
             aveData_new,
             color="skyblue",
             ls='-.',
             label=r"$\lambda=0.8$")
    x_axis_new, aveData_new = dealxy(x_axis, dealData(aveErrors[3]))
    plt.plot(x_axis_new,
             aveData_new,
             color="blue",
             ls=':',
             label=r"$\lambda=0.9$")

    plt.legend()
    plt.xlabel(r"$\alpha$")
    plt.ylabel("平均均方根误差")
    plt.show()


def nStepTDLambdaLearningExample(agent, env):
    ''' 取不同学习率进行实验 '''
    x = np.linspace(0.01, 1, 100)

    for alpha_num in x:  # 不同lambda取值的实验结果
        for lambda_ in lambda_dict.keys():
            for _ in range(nums_for_ave):
                agent.nStepTDLambdaLearning(env=env,
                                            lambda_=lambda_,
                                            gamma=1,
                                            alpha=alpha_num,
                                            max_episode_num=6)

    plot_lambda_alpha(agent.aveErrors)


if __name__ == "__main__":
    agent = Agent()
    env = RandomWalkEnv()  # 引入环境env
    print("Learning...")
    nStepTDLambdaLearningExample(agent, env)
