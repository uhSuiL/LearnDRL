深度强化学习——原理、算法与PyTorch实战，代码名称：代17-例6.1-基于Sarsa算法的确定环境扫地机器人问题的Q值迭代.py
#Sarsa算法解决确定环境扫地机器人代码
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

def policy_epsilon_greedy(s, Q, epsilon):#ε贪心策略
    Q_s = Q[s]
    action = vilid_action_space(s)
    if np.random.rand() < epsilon:
        a = np.random.choice(action)
    else:
        index_a = np.argmax([Q_s[i] for i in action])
        a = action[index_a]
    return a

def print_dd(s, a, next_s, next_a, print_len, episode_i, Q,e_k,a_k):
    for i in range(1):  
        if episode_i == int(print_len * (0.1 * i + 1)):
            if s == 21 and a == 1 and next_s == 22 and next_a == 1:
                print("*********************************单步的计算过程**************************************")
                print("alpha:"+str(a_k))
                print("epsilon:"+str(e_k))
                print("state:" + str(int(print_len * (0.1 * i + 1))))
                print(Q[s][a])
                print(Q[next_s][a])
                print("update:"+str(Q[s, a] + a_k * (0.8 * Q[next_s, next_a] - Q[s, a])))
                print("************************************************************************************")

def trans(Q_S):#因为环境中动作顺序是左右上下，文章建模的动作顺序是上下左右，所以转换为文章中的顺序（上下左右）进行输出，并保存3位小数
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

def Attenuation(epsilon,alpha,episode_sum,episode):#epsilon和alpha衰减函数
    epsilon = (float(episode_sum) - float(episode)) / float(episode_sum) * epsilon
    alpha = (float(episode_sum) - float(episode)) / float(episode_sum) * alpha
    return epsilon, alpha

def Sarsa(env, episode_num, alpha, gamma, epsilon):
    Q = np.zeros((env.n_width * env.n_height, env.action_space.n))
    list_q = [0,1,2,3,4,2998,2999,3000,3001,3002,19996,19997,19998,19999,20000]
    for episode_i in range(episode_num):
        env.reset()
        S = env.state
        epsilon_k, alpha_k = Attenuation(epsilon,alpha,episode_num,episode_i)
        A = policy_epsilon_greedy(S, Q, epsilon_k)
        done = False
        print_ff(list_q,Q,episode_i,epsilon_k,alpha_k)
        while not done:
            next_S, R, done, _ = env.step(A)
            next_A = policy_epsilon_greedy(next_S, Q, epsilon_k)
            print_dd(S,A,next_S,next_A,3000,episode_i,Q,epsilon_k,alpha_k)
            Q[S, A] = Q[S, A] + alpha_k * (R + gamma * Q[next_S, next_A] - Q[S, A])
            S = next_S
            A = next_A            
    return Q

Q = Sarsa(env, 20000, 0.05, 0.8, 0.5)