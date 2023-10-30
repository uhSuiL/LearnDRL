深度强化学习——原理、算法与PyTorch实战，代码名称：代20-例6.4-基于期望Sarsa算法的确定环境扫地机器人问题的Q值迭代.py
#期望Sarsa算法
from book_gridword import GridWorldEnv
import numpy as np
from queue import Queue
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

def compute_epsion(s,Q,epsilon):
    max_a = np.argmax(Q[s])
    action = vilid_action_space(s)
    len_all_a = len(action)
    prob_l = [0.0,0.0,0.0,0.0]
    for index_a in action:
        if index_a == max_a:
            prob_l[index_a] = 1.0 - epsilon + (epsilon / len_all_a)
        else:
            prob_l[index_a] = epsilon / len_all_a
    return prob_l

def compute_e_q(prob, q_n):
    sum = 0.0
    for i in range(4):
        sum += prob[i] * q_n[i]
    return sum

def trans1(Q_S):
    new_Q = []
    new_Q.append(Q_S[2])
    new_Q.append(Q_S[3])
    new_Q.append(Q_S[0])
    new_Q.append(Q_S[1])
    return new_Q

def print_dd(s, a, next_s, print_len, episode_i, Q, e_k, a_k):
    for i in range(50):  
        if episode_i == int(print_len * ((0.02 * i)+1)):
            if s == 15 and a == 3 and next_s == 10:
                print("*****************************单步计算过程****************************************")
                print("alpha:"+str(a_k))
                print("epsilon:"+str(e_k))
                print("state:" + str(int(print_len * (1 + (0.02 * i)))))
                print("Q(%d,%d)"%(s,a))
                print(Q[s][a])
                print("Q(%d,*)"%(next_s))
                print(trans1(Q[next_s]))
                prob_l = compute_epsion(next_s, Q, e_k)
                print('概率'+ str(trans1(prob_l)))
                Q_e = compute_e_q(prob_l,Q[next_s])
                print('update:'+str(Q[s, a] + a_k * (0.8 * Q_e - Q[s, a])))

def trans(Q_S):
    new_Q = []
    new_Q.append(round(Q_S[2],3))
    new_Q.append(round(Q_S[3],3))
    new_Q.append(round(Q_S[0],3))
    new_Q.append(round(Q_S[1],3))
    return new_Q

def print_ff(list_q, Q, episode_i,epsilon_k,alpha_k):
    list_s = range(0,25)
    for em in list_q:
        if em == episode_i:
            print("*******************************情节数:%s*******************************"%(str(em)))
            for state in list_s:
                print("Q(%d,*) "%(state) + str(trans(Q[state])))
                action = vilid_action_space(state)
                len_a = len(action)
                e_p = epsilon_k / float(len_a)
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

def  Expectation_sarsa(env, episode_num, alpha, gamma, epsilon):
    Q = np.zeros((env.n_width * env.n_height, env.action_space.n))
    Q_queue = Queue(maxsize=11)
    list_q = [0,1,2,3,4,5998,5999,6000,6001,6002,19996,19997,19998,19999,20000]
    for episode_i in range(episode_num):
        env.reset()
        s = env.state
        epsilon_k, alpha_k = Attenuation(epsilon,alpha,episode_num,episode_i)
        print_ff(list_q, Q, episode_i,epsilon_k,alpha_k)
        done = False
        while not done:
            a = policy_epsilon_greedy(s, Q, epsilon_k)
            next_s, r, done, _ = env.step(a)
            print_dd(s, a, next_s, 6000, episode_i, Q, epsilon_k,alpha_k)
            prob_l = compute_epsion(next_s, Q, epsilon_k)
            Q_e = compute_e_q(prob_l,Q[next_s])
            Q[s, a] += alpha_k * (r + gamma * Q_e - Q[s, a])
            s = next_s
    return Q

Q = Expectation_sarsa(env,20000, 0.05, 0.8, 0.5)