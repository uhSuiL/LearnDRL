''' 深度强化学习——原理、算法与PyTorch实战，代码名称：代25-例7.4 Sarsa(0)、n-步Sarsa以及TD(λ)算法应用于确定环境扫地机器人问题中的对比分析.py '''

# 导入对应的模块
from myBookEnvCode.my_book_gridworld0406 import GridWorldEnv
import numpy as np
import pandas as pd
from IPython.display import display


class nStepSarsaLearningAgent():
    def __init__(self):
        self.Q = {}  # {state:{action:q_value}} 初始化Q ★

    def create_epsilon_greedy_policy(self, nA):
        """
        Creates an epsilon-greedy policy based on Q values.

        Args:
            Q: A dictionary that maps from state -> action values

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        def policy_fn(action_list, state, epsilon):
            A = np.zeros(nA, dtype=float)
            valid_nA = len(action_list[state])
            for action in action_list[state]:
                A[action] = epsilon / valid_nA  # 在探索阶段，每个动作都有epsilon / valid_nA的概率被选到
            best_action = max(
                self.Q[state],
                key=self.Q[state].get)  # 获取最大value值对应的key，即得到最大动作值函数对应的动作
            A[best_action] += 1.0 - epsilon  # 在利用阶段，最大动作值函数对应的动作另外有1.0 - epsilon的概率被选到
            return A

        return policy_fn

    def nStepSarsaLearning(self, env, n, gamma, max_episode_num):
        ''' n步Sarsa算法学习 '''
        target_policy = self.create_epsilon_greedy_policy(
            env.action_space.n)  # 根据动作数创建一个策略
        num_episode = 0
        for state in range(env.observation_space.n):  # 初始化各状态的Q
            self.initValue(state,
                           env.valid_actions_for_states,
                           randomized=False)
        self.initPrintTable()  #初始化保存输出结果的表
        ''' 运行max_episode_num情节 '''
        while num_episode < max_episode_num:
            s = env.reset()  #重置环境
            ''' 根据策略选择一个动作 '''
            probs = target_policy(env.valid_actions_for_states, s,
                                  epsilon_by_epsiode(num_episode))
            a = np.random.choice(np.arange(len(probs)), p=probs)

            if num_episode == 0:
                self.saveEps(env, target_policy, num_episode)  #保存初始状态
            time_in_episode = 0
            T_ep = 1000000
            ep_states = [s]
            ep_actions = [a]
            ep_rewards = [0]  #奖赏下标从1开始
            time_update = 0
            while True:
                if time_in_episode < T_ep:
                    n_s, n_r, done, _ = env.step(a)  # 进入下一个状态
                    if not self.__isStateInQ(n_s):  #初始化状态值
                        self.initValue(n_s,
                                       env.valid_actions_for_states,
                                       randomized=False)
                    # 保存奖赏和状态
                    ep_states.append(n_s)
                    ep_rewards.append(n_r)

                    if done:
                        T_ep = time_in_episode + 1
                    else:
                        probs = target_policy(env.valid_actions_for_states,
                                              n_s,
                                              epsilon_by_epsiode(num_episode))
                        n_a = np.random.choice(np.arange(len(probs)), p=probs)
                        ep_actions.append(n_a)
                        s = n_s
                        a = n_a
                time_update = time_in_episode - n + 1
                if time_update >= 0:
                    n_step_return = 0
                    xs = 1
                    for i in range(time_update + 1,
                                   min(time_update + n + 1, T_ep + 1)):
                        n_step_return += xs * ep_rewards[i]
                        xs *= gamma
                    if (time_update + n < T_ep):
                        n_step_return += xs * self.__getQValue(
                            ep_states[time_update + n],
                            ep_actions[time_update + n])
                    ''' 获取旧的Q值，更新后写进Q表 '''
                    old_q = self.__getQValue(ep_states[time_update],
                                             ep_actions[time_update])
                    new_q = old_q + alpha_by_epsiode(num_episode) * (
                        n_step_return - old_q)
                    self.__setQValue(ep_states[time_update],
                                     ep_actions[time_update], new_q)

                time_in_episode += 1
                if (time_update == T_ep - 1):
                    break

            num_episode += 1
            if num_episode in self.printEps:
                self.saveEps(env, target_policy, num_episode)

        display(self.printTable)  # 打印表格
        self.printTable.to_excel('15-表7.5-n-步Sarsa(n=3)算法更新过程表.xlsx')  #保存excel

    def saveEps(self, env, target_policy, num_episode):
        ''' 保存一个情节信息 '''
        Q_s, Probs_s = [], []  # 保存一个情节所有状态的Q和策略
        for s in self.Q.keys():  # 遍历所有状态
            Q_sa, prob_sa = '', ''  # 保存一个状态所有动作的值和策略
            if s != 12:
                for a in range(4):  # 遍历所有动作
                    Q_sa += format(self.Q[s][a],
                                   '.3f') if a in self.Q[s].keys() else '*.***'
                    if a != 3:
                        Q_sa += ';'
                Q_s.append(Q_sa)
                probs = target_policy(env.valid_actions_for_states, s,
                                      epsilon_by_epsiode(num_episode))
                for a in range(4):
                    prob_sa += format(probs[a], '.3f')
                    if a != 3:
                        prob_sa += ';'
                Probs_s.append(prob_sa)
        self.printTable.loc['Q' + str(num_episode)] = Q_s  #保存该情节的所有Q值
        self.printTable.loc[chr(960) + str(num_episode)] = Probs_s  #保存该情节的所有策略

    def initPrintTable(self):
        ''' 初始化保存输出结果的表 '''
        self.printEps = [
            1, 2, 3, 4, 7498, 7499, 7500, 7501, 7502, 12498, 12499, 12500,
            12501, 12502, 19970, 19971, 19972, 19973, 19974, 19975, 19996,
            19997, 19998, 19999, 20000
        ]  # 待打印的动作，包括初始情节，中间情节，收敛情节和结束时的情节
        self.printState = ['S' + str(i) for i in range(25)]  # 待打印的状态，除开S12
        self.printState.remove('S12')
        self.printTable = pd.DataFrame(columns=self.printState)

    # if a state exists in Q dictionary
    def __isStateInQ(self, state):
        # 判断空值。有值则返回值，无值则返回None - None is not None = False
        return self.Q.get(state) is not None  # 因为是实例属性，所以要用self.进行引用

    def initValue(self, s, valid_actions_list, randomized=False):  # 初始化Q和E
        # Q[s]为空值时进入判断
        if not self.__isStateInQ(s):
            self.Q[s] = {}  # 初始化Q
            for a in valid_actions_list[s]:  # 遍历所有action_name
                self.Q[s][a] = np.random.random(
                ) / 10 if randomized is True else 0.0  # 初始化Q(s,a)；随机一个动作值函数。只有结束状态的Q(s,a) = 0

    def _state_to_xy(self, s):
        ''' 实现状态号到格子上坐标的转换 '''
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    """Q与E的获取与设置方法"""

    def __getQValue(self, s, a):  # ①
        ''' Q值的获取 '''
        return self.Q[s][a]

    def __setQValue(self, s, a, new_q):  # ②
        ''' Q值的设置 '''
        self.Q[s][a] = new_q


class sarsaLambdaAgent():
    def __init__(self):
        self.Q = {}  # {state:{action:q_value}} 初始化Q ★
        self.E = {
        }  # {state:{action:E_value}} 初始化Eligibility Trace（适合度轨迹），用于Sarsa(λ) ★

    def create_epsilon_greedy_policy(self, nA):
        """
        Creates an epsilon-greedy policy based on Q values.

        Args:
            Q: A dictionary that maps from state -> action values

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        def policy_fn(action_list, state, epsilon):
            A = np.zeros(nA, dtype=float)
            valid_nA = len(action_list[state])
            for action in action_list[state]:
                A[action] = epsilon / valid_nA  # 在探索阶段，每个动作都有epsilon / valid_nA的概率被选到
            best_action = max(
                self.Q[state],
                key=self.Q[state].get)  # 获取最大value值对应的key，即得到最大动作值函数对应的动作
            A[best_action] += 1.0 - epsilon  # 在利用阶段，最大动作值函数对应的动作另外有1.0 - epsilon的概率被选到
            return A

        return policy_fn

    def sarsaLambdaLearning(self, env, lambda_, gamma, max_episode_num):
        target_policy = self.create_epsilon_greedy_policy(env.action_space.n)
        num_episode = 0
        for state in range(env.observation_space.n):
            self.initValue(state,
                           env.valid_actions_for_states,
                           randomized=False)
        self.initPrintTable()  #初始化保存输出结果的表
        while num_episode <= max_episode_num:
            s = env.reset()
            probs = target_policy(env.valid_actions_for_states, s,
                                  epsilon_by_epsiode(num_episode))
            a = np.random.choice(np.arange(len(probs)), p=probs)
            self.__resetEValue()  # 比SARSA(0)多了要初始化E
            if num_episode == 0:
                self.saveEps(env, target_policy, num_episode)  #保存初始状态

            time_in_episode = 0
            while True:
                n_s, n_r, done, _ = env.step(a)  # 进入一下个状态
                if not self.__isStateInQ(n_s):
                    self.initValue(n_s,
                                   env.valid_actions_for_states,
                                   randomized=False)
                if not done:
                    probs = target_policy(env.valid_actions_for_states, n_s,
                                          epsilon_by_epsiode(num_episode))
                    n_a = np.random.choice(np.arange(len(probs)), p=probs)
                q = self.__getQValue(s, a)  # 取Q(s,a)
                if not done:
                    q_next = self.__getQValue(n_s, n_a)
                else:
                    q_next = 0
                delta = n_r + gamma * q_next - q

                e = self.__getEValue(s, a)  # 得到E值
                e = e + 1  # 每经历一个(s,a)，就为这个(s,a)下的E(s,a) 增加1
                self.__setEValue(
                    s, a, e
                )  # 存下当前(s,a)的E。因为走一步就会有一个E，所以要先把该(s,a)存入，后面的遍历才可以访问到该(s,a)

                # SARSA(λ)，遍历所有经历过的(s,a)
                # 回溯之前的Q。不讲究顺序。每前进一次，所有存在过的(s,a)都会被λ处理一次，出现的越早的(s,a)就会被alpha*e_value乘的越多，有值状态的更新就会表的越微小
                for s, action_e_dic in list(zip(self.E.keys(),
                                                self.E.values())):  # 遍历所有state
                    for action_name, e_value in list(
                            zip(action_e_dic.keys(),
                                action_e_dic.values())):  # 遍历每个state下的action
                        ''' 获取旧的Q值，更新后写进Q表 '''
                        old_q = self.__getQValue(s, action_name)
                        new_q = old_q + alpha_by_epsiode(
                            num_episode) * delta * e_value  # alpha 步长
                        new_e = gamma * lambda_ * e_value
                        self.__setQValue(s, action_name, new_q)  # 更新历史(s,a)的Q
                        self.__setEValue(s, action_name, new_e)  # 更新

                s = n_s

                time_in_episode += 1
                if done:
                    break
                else:
                    a = n_a

            num_episode += 1
            if num_episode in self.printEps:
                self.saveEps(env, target_policy, num_episode)
        pd.set_option('display.max_rows', 500)
        display(self.printTable)  # 打印表格
        self.printTable.to_excel(
            '16-表7.6-Sarsa(λ)(λ=0.8)算法更新过程表.xlsx')  #保存excel

    def saveEps(self, env, target_policy, num_episode):
        ''' 保存一个情节信息 '''
        Q_s, E_s, Probs_s = [], [], []  # 保存一个情节所有状态的Q,E和策略
        for s in self.Q.keys():  # 遍历所有状态
            Q_sa, E_sa, prob_sa = '', '', ''  # 保存一个状态所有动作的值和策略
            if s != 12:
                for a in range(4):  # 遍历所有动作
                    Q_sa += format(self.Q[s][a],
                                   '.3f') if a in self.Q[s].keys() else '*.***'
                    E_sa += format(self.E[s][a],
                                   '.3f') if a in self.E[s].keys() else '*.***'
                    if a != 3:
                        Q_sa += ';'
                        E_sa += ';'
                Q_s.append(Q_sa)
                E_s.append(E_sa)
                probs = target_policy(env.valid_actions_for_states, s,
                                      epsilon_by_epsiode(num_episode))
                for a in range(4):
                    prob_sa += format(probs[a], '.3f')
                    if a != 3:
                        prob_sa += ';'
                Probs_s.append(prob_sa)
        self.printTable.loc['Q' + str(num_episode)] = Q_s  #保存该情节的所有Q值
        self.printTable.loc['E' + str(num_episode)] = E_s  #保存该情节的所有E值
        self.printTable.loc[chr(960) + str(num_episode)] = Probs_s  #保存该情节的所有策略

    def initPrintTable(self):
        ''' 初始化保存输出结果的表 '''
        self.printEps = [
            1, 2, 3, 4, 7498, 7499, 7500, 7501, 7502, 12498, 12499, 12500,
            12501, 12502, 19970, 19971, 19972, 19973, 19974, 19975, 19996,
            19997, 19998, 19999, 20000
        ]  # 待打印的动作，包括初始情节，中间情节，收敛情节和结束时的情节
        self.printState = ['S' + str(i) for i in range(25)]  # 待打印的状态，除开S12
        self.printState.remove('S12')
        self.printTable = pd.DataFrame(columns=self.printState)

    def __isStateInQ(self, state):
        # 判断空值。有值则返回值，无值则返回None - None is not None = False
        return self.Q.get(state) is not None  # 因为是实例属性，所以要用self.进行引用

    def initValue(self, s, valid_actions_list, randomized=False):  # 初始化Q和E
        # Q[s]为空值时进入判断
        if not self.__isStateInQ(s):
            self.Q[s] = {}  # 初始化Q
            self.E[s] = {}
            for a in valid_actions_list[s]:  # 遍历所有action_name
                self.Q[s][a] = np.random.random(
                ) / 10 if randomized is True else 0.0  # 初始化Q(s,a)；随机一个动作值函数。只有结束状态的Q(s,a) = 0
                self.E[s][a] = 0

    """Q与E的获取与设置方法"""

    def __getQValue(self, s, a):  # ①
        ''' Q值的获取 '''
        return self.Q[s][a]

    def __setQValue(self, s, a, new_q):  # ②
        ''' Q值的设置 '''
        self.Q[s][a] = new_q

    def __getEValue(self, s, a):
        ''' E值的获取 '''
        return self.E[s][a]

    def __setEValue(self, s, a, new_q):
        ''' E值的设置 '''
        self.E[s][a] = new_q

    def __resetEValue(self):
        ''' E值的重置 '''
        for action_Evalue in self.E.values():
            for action in action_Evalue.keys():
                action_Evalue[action] = 0.00


def nStepSarsaLearningExample(agent, env):
    agent.nStepSarsaLearning(env=env, n=3, gamma=0.8, max_episode_num=20000)


def sarsaLambdaLearningExample(agent, env):
    agent.sarsaLambdaLearning(env=env,
                              lambda_=0.8,
                              gamma=0.8,
                              max_episode_num=20000)


if __name__ == "__main__":
    ''' 设置实验的参数 '''
    np.random.seed(1)
    epsilon_start = 0.5
    epsilon_final = 0
    epsilon_episodes = 20000
    alpha_start = 0.05
    alpha_final = 0
    alpha_episodes = 20000
    epsilon_by_epsiode = lambda episode_idx: epsilon_start - (
        epsilon_start - epsilon_final) * min(episode_idx, epsilon_episodes
                                             ) / epsilon_episodes
    alpha_by_epsiode = lambda episode_idx: alpha_start - (
        alpha_start - alpha_final) * min(episode_idx, alpha_episodes
                                         ) / alpha_episodes

    env = GridWorldEnv()  # 引入环境env
    nStepSarsaLearningagent = nStepSarsaLearningAgent()
    sarsaLambdaLearningagent = sarsaLambdaAgent()

    print("nStepSarsa Learning...")
    nStepSarsaLearningExample(nStepSarsaLearningagent, env)
    print("sarsaLambda Learning...")
    sarsaLambdaLearningExample(sarsaLambdaLearningagent, env)