深度强化学习——原理、算法与PyTorch实战，代码名称：代19-例6.3-基于Q-Learning算法的确定环境扫地机器人问题的Q值迭代.py
#Q-learning算法
from book_gridword import GridWorldEnv
import numpy as np
np.random.seed(1)
env = GridWorldEnv()
#有效动作空间
def vilid_action_space(s):
    action_sacpe = []
    if s % 5 != 0:#左
        action_sacpe.append(0)
    if s % 5 != 4:#右
        action_sacpe.append(1)
    if s <= 19:#上
        action_sacpe.append(2)
    if s >= 5:#下
        action_sacpe.append(3)
    return action_sacpe

def policy_epsilon_greedy(s, Q, epsilon):
    Q_s = Q[s]
    action = vilid_action_space(s)
    if np.random.rand() < epsilon:
        a = np.random.choice(action)
    else:
        index_a = np.argmax([Q_s[i] for i in action])
        a = action[index_a]
    return a

def trans1(Q_S):
    new_Q = []
    new_Q.append(Q_S[2])
    new_Q.append(Q_S[3])
    new_Q.append(Q_S[0])
    new_Q.append(Q_S[1])
    return new_Q

def trans(Q_S):
    new_Q = []
    new_Q.append(round(Q_S[2],3))
    new_Q.append(round(Q_S[3],3))
    new_Q.append(round(Q_S[0],3))
    new_Q.append(round(Q_S[1],3))
    return new_Q

def print_dd(s, a, next_s, print_len, episode_i, Q,e_k,a_k):
    for i in range(2):  
        if episode_i == int(print_len * (0.1 * i + 1)):
            if s == 15 and a == 3 and next_s == 10:
                print("*********************************单步的计算过程***************************************")
                print("alpha:"+str(a_k))
                print("epsilon:"+str(e_k))
                print("state:" + str(int(print_len * (0.1 * i + 1))))
                print("Q(%d,%d)"%(s,a))
                print(Q[s][a])
                print("Q(%d,*)"%(next_s))
                print(trans1(Q[next_s]))
                print('output:'+str(Q[s][a] + a_k * (0.8 * np.max(Q[next_s]) - Q[s, a])))

def print_ff(list_q, Q, episode_i,epsilon_k,alpha_k):
    list_s = range(0,25)
    for em in list_q:
        if em == episode_i:
            print("*******************************情节数:%s*******************************"%(str(em)))
            for state in list_s:
                print("Q(%d,*)"%(state) + str(trans(Q[state])))
                action = vilid_action_space(state)
                len_a = len(action)
                e_p = epsilon_k / float(len_a)
                max_a = np.argmax(Q[state])
                prob = []
                index_a = np.argmax([Q[state][i] for i in action])
                for i in range(4):#计算epsilon
                    if i not in action:
                        prob.append(0.0)
                    else:
                        if i == action[index_a]:
                            prob.append(1 - epsilon_k + e_p)
                        else:
                            prob.append(e_p)
                print('概率值:' + str(trans(prob)))
                print("epsilon_k: {}".format(epsilon_k))
                print("alpha_k:{}".format(alpha_k))

def Attenuation(epsilon,alpha,episode_sum,episode):
    epsilon = (float(episode_sum) - float(episode)) / float(episode_sum) * epsilon
    alpha = (float(episode_sum) - float(episode)) / float(episode_sum) * alpha
    return epsilon, alpha

def Q_Learning(env, episode_num, alpha, gamma, epsilon):
    Q = np.zeros((env.n_width * env.n_height, env.action_space.n))
    list_q = [0,1,2,3,4,10997,10998,10999,11000,11001,24996,24997,24998,24999,25000]
    for episode_i in range(episode_num):
        env.reset()
        s = env.state
        epsilon_k, alpha_k = Attenuation(epsilon,alpha,episode_num,episode_i)
        print_ff(list_q, Q, episode_i,epsilon_k,alpha_k)
        done = False
        while not done:
            a = policy_epsilon_greedy(s, Q, epsilon_k)
            next_s, r, done, _ = env.step(a)
            print_dd(s, a, next_s, 10000, episode_i, Q, epsilon_k, alpha_k)
            Q[s, a] += alpha_k * (r + gamma * np.max(Q[next_s]) - Q[s, a])
            s = next_s
    return Q

Q = Q_Learning(env, 25000, 0.05, 0.8, 0.5)