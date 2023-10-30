深度强化学习——原理、算法与PyTorch实战，代码名称：代21-例6.5-基于Double Q-Learning算法的确定环境扫地机器人问题的Q值迭代.py
#Double Q_learning算法
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

def policy_epsilon_greedy(s, Q1, Q2, epsilon):
    Q_s = Q1[s] + Q2[s]
    action = vilid_action_space(s)
    if np.random.rand() < epsilon:
        a = np.random.choice(action)
    else:
        index_a = np.argmax([Q_s[i] for i in action])
        a = action[index_a]
    return a

def print_dd(s, a, next_s, print_len, episode_i, Q1, Q2, e_k, a_k):  
    if episode_i == 8640:
        if s == 15 and a == 3 and next_s == 10:
            print("********************************单步计算过程************************************")
            print("alpha:"+str(a_k))
            print("epsilon:"+str(e_k))
            print(trans1(Q1[next_s]))
            print(trans1(Q1[s]))
            print(trans1(Q2[next_s]))
            print('update:'+str(Q1[s][a]+a_k * (0.8 * Q2[next_s,np.argmax(Q1[next_s])] - Q1[s,a])))

def print_ff(list_q, Q1,Q2, episode_i,epsilon_k, alpha_k):
    list_s = range(0,25)
    for em in list_q:
        if em == episode_i:
            print('*******************************情节数:%s**************************'%(str(em)))
            for state in list_s:
                print("Q1(%d,*)"%(state) + str(trans(Q1[state])))
                print("Q2(%d,*)"%(state) + str(trans(Q2[state])))
                action = vilid_action_space(state)
                len_a = len(action)
                e_p = epsilon_k / float(len_a)
                max_a = np.argmax(Q1[state]+Q2[state])
                prob = []
                Q = Q1[state]+Q2[state]
                index_a = np.argmax([Q[i] for i in action])
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

def double_learning(env, episode_num, alpha, gamma, epsilon):
    Q1 = np.zeros((env.n_width * env.n_height, env.action_space.n))
    Q2 = np.zeros((env.n_width * env.n_height, env.action_space.n))
    list_q = [0,1,2,3,4,8638,8639,8640,8641,8642,39996,39997,39998,39999,40000]
    for episode_i in range(episode_num):
        env.reset()
        s = env.state
        epsilon_k, alpha_k = Attenuation(epsilon,alpha,episode_num,episode_i)
        done = False
        print_ff(list_q, Q1, Q2, episode_i, epsilon_k, alpha_k)
        while not done:
            a = policy_epsilon_greedy(s, Q1, Q2, epsilon_k)
            next_s, r, done, _ = env.step(a)
            pros = np.random.rand()
            if pros < 0.5:
                print_dd(s, a, next_s, 8000, episode_i, Q1, Q2, epsilon_k, alpha_k)
                Q1[s,a] += alpha_k * (r + gamma * Q2[next_s,np.argmax(Q1[next_s])] - Q1[s,a])
            else:
                Q2[s,a] += alpha_k * (r + gamma * Q1[next_s,np.argmax(Q2[next_s])] - Q2[s,a])
            s = next_s        
    return (Q1 + Q2) / 2.0

Q = double_learning(env, 40000, 0.05, 0.8, 0.5)