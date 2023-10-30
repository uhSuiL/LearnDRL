深度强化学习——原理、算法与PyTorch实战，代码名称：代18-例6.2-风险投资问题.py
# Sarsa算法解决风险投资问题代码
import numpy as np
from invest import InvestEnv
np.random.seed(1)
env = InvestEnv()

def trans_q(Q):  # 输出保留3位小数
    new_q = []
    new_q = [round(x, 3) for x in Q]
    return new_q

def Attenuation(epsilon, alpha, episode_sum, episode):  # epsilon和alpha衰减函数
    epsilon = (float(episode_sum) - float(episode)) / float(episode_sum) * epsilon
    alpha = (float(episode_sum) - float(episode)) / float(episode_sum) * alpha
    return epsilon, alpha

# 输出函数
def print_ff(list_q, Q, episode_i, epsilon_k, alpha_k):
    list_s = range(0,50)
    for em in list_q:
        if em == episode_i:
            print("*******************************情节数:%s*******************************" % (str(em)))
            for state in list_s:
                print("Q(%d,*)" % (state) + str(trans_q(Q[state])))
                prob = [epsilon_k / 2.0, epsilon_k / 2.0]
                max_a = np.argmax(Q[state])
                prob[max_a] = 1 - (epsilon_k / 2.0)
                print('概率值' + str(trans_q(prob)))
                print("epsilon_k: {}".format(epsilon_k))
                print("alpha_k:{}".format(alpha_k))

# 输出单步计算过程
def print_dd(s, a, R, next_s, next_a, print_len, episode_i, Q, e_k, a_k, P, P_next):

    if s == 6 and a == 1:
        print("*********************************单步的计算过程************************************")
        print(6, 1)
        print("alpha:" + str(a_k))
        print("epsilon:" + str(e_k))
        print("Q_state: {} Q_next: {}".format(Q[s], Q[next_s]))
        print("Q[{},{}]: {}".format(s, a, Q[s, a]))
        print("Q[{},{}]: {}".format(next_s, next_a, Q[next_s, next_a]))

        print("update:" + str(Q[s, a] + a_k * (R + 0.8 * Q[next_s, next_a] - Q[s, a])))
        # print(p)
        print("************************************************************************************")

def policy_epsilon_greedy(env, s, Q, epsilon):
    Q_s = Q[s]
    if np.random.rand() < epsilon:
        a = np.random.choice(env.action_space)
    else:
        a = np.argmax(Q_s)
    return a

def Sarsa(env, episode_num, alpha, gamma, epsilon):
    Q = np.zeros((env.state_space, env.action_space))
    epsilon = epsilon
    count = 0
    list_q = [0,1,2,3,4,9998,9999,10000,10001,10002,19996,19997,19998,19999,20000]
    for episode_i in range(episode_num):
        env.reset()
        S = env.state
        epsilon_k, alpha_k = Attenuation(epsilon, alpha, episode_num, episode_i)
        A = policy_epsilon_greedy(env, S, Q, epsilon_k)
        print_ff(list_q, Q, episode_i, epsilon_k, alpha_k)
        if episode_i == 10000:
            print("e_k:" + str(epsilon_k) + "a_k" + str(alpha_k))
        done = False
        P_S = env.getState()
        for i in range(1000):
            next_S, R = env.step(A)
            P_S_next = env.getState()
            if next_S > 49:
                Q[S, A] = Q[S, A] + alpha_k * (R + gamma * 0.0 - Q[S, A])
                break
            next_A = policy_epsilon_greedy(env, next_S, Q, epsilon_k)
            # 输出某一个
            if episode_i in [9999, 10000]:
                print_dd(S, A, R, next_S, next_A, 10000, episode_i, Q, epsilon_k, alpha_k, P_S, P_S_next)
            Q[S, A] = Q[S, A] + alpha_k * (R + gamma * Q[next_S, next_A] - Q[S, A])
            S = next_S
            A = next_A
            P_S = P_S_next
        if S > 10.0:
            count += 1
    print(count)
    return Q


Q = Sarsa(env, 20001, 0.05, 0.8, 0.5)