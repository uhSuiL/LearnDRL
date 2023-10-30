# 代08-例4.4-基于动作值函数的随机环境扫地机器人任务策略评估
import numpy as np
import pandas as pd

"""定义格子世界参数"""
world_h =  5
world_w = 5
length = world_h * world_w
gamma = 0.8
action = ['n', 's', 'w', 'e']  # 动作名称
ds_action = {'n': -world_w, 'e': 1, 's': world_w, 'w': -1}
policy = np.zeros([length, len(action)])
suqe=[20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3,4]



# 定义奖励
def reward(s):
    if s == 20:  # 到充电站
        return 1
    elif s == 12:  # 到陷阱中
        return -10
    elif s == 9:  # 到垃圾处
        return 3
    else:
        return 0  # 其他
    # in表示0是[*，*，*]中的一个


def getAction(a):
    if a == 'n':
        return 0
    elif a == 'e':
        return 3
    elif a == 's':
        return 1
    elif a == 'w':
        return 2


# 在s状态下执行动作a，返回下一状态（编号）
def next_states(s, a):
    # 越过边界时pass
    if (s < world_w and a == 'n') \
            or (s % world_w == 0 and a == 'w') \
            or (s > length - world_w - 1 and a == 's') \
            or ((s + 1) % world_w == 0 and a == 'e'):  # (s % (world_w - 1) == 0 and a == 'e' and s != 0)
        next_state = s  # 表现为next_state不变
    else:
        next_state = s + ds_action[a]  # 进入下一个状态
    return next_state


# 在s状态下执行动作，返回所有可能的下一状态（编号）list
def getsuccessor(s):
    successor = []
    for a in action:  # 遍历四个动作
        if s == next_states(s, a):
            continue
        else:
            # print("状态s=%s,动作a=%s"%(s,a))
            next = next_states(s, a)  # 得到下一个状态（编号）
        successor.append(next)  # 以list保存当前状态s下执行四个动作的下一状态
    # print(len(successor))
    return successor

def envActionPolicy(a):
    if a == 'n':
        return 's'
    elif a == 's':
        return 'n'
    elif a == 'e':
        return 'w'
    elif a == 'w':
        return 'e'

def CaValue(Q):
    v = [0 for i in range(length)]
    for i in range(length):
        for a in action:
            newAction = getAction(a)
            v[i] += policy[i][newAction] * Q.loc[i, a]
    value = np.array(v).reshape(world_h, world_w)
    print(np.round(value, decimals=4))


def sumQ_nextstate(s, Q, visios):
    sum = 0
    for i in action:
        newAction = getAction(i)
        sum += policy[s][newAction] * Q.loc[s, i]

    return sum

def initPolicy():
    for s in range(length):
        for a in action:
            if next_states(s, a) == s:
                continue
            newAction = getAction(a)
            policy[s][newAction] = 1 / len(getsuccessor(s))
    # print(policy)
def policy_eval_Q_random(theta=0.0001):
    Q = pd.DataFrame(
        np.zeros((length, len(action))),  # q_table initial values
        columns=action,  # actions's name
    )

    iter = 0
    while True:
        k = -1
        delta = 0  # 定义最大差值，判断是否有进行更新
        for s in suqe:  # 遍历所有状态 [0~25]
            visio = False
            k += 1
            if s in [9, 20, 12]:  # 若当前状态为吸入状态，则直接pass不做操作
                continue
            if s == 17:
                visio = True
            # [[-0.7954 - 1.0218 - 1.2655 - 0.1564  1.369]
            #  [-1.066 - 1.9614 - 3.8893 - 0.7455  0.]
            # [-1.4346 - 4.176
            # 0. - 3.5631 - 0.0563]
            # [-0.489 - 1.7904 - 4.1252 - 1.7891 - 0.6118]
            # [0. - 0.4778 - 1.3917 - 0.9611 - 0.5992]]
            for a in action:
                newAction = getAction(a)
                env_action = envActionPolicy(a)
                next_state = next_states(s, a)
                env_state = next_states(s, env_action)
                rewards = reward(next_state)
                env_rewards = reward(env_state)
                if policy[s][newAction] == 0:
                    continue
                if next_state == 12:
                    q = 0.8 * (rewards + gamma * sumQ_nextstate(s, Q, visio)) + 0.15 * (
                                gamma * sumQ_nextstate(s, Q, visio)) + 0.05 * (
                                    env_rewards + gamma * (sumQ_nextstate(env_state, Q, visio)))
                    if visio == True:
                        print("q=%.2f=0.8*(%.2f+%.2f*%.2f)+0.15*(%.2f*%.2f)+0.05*(%.2f+%.2f*%.2f)"
                              % (q, rewards, gamma, sumQ_nextstate(s, Q, visio), gamma, sumQ_nextstate(s, Q, visio),
                                 env_rewards, gamma, sumQ_nextstate(env_state, Q, visio)))
                else:
                    q = 0.8 * (rewards + gamma * sumQ_nextstate(next_state, Q, visio)) + 0.05 * (
                                env_rewards + gamma * sumQ_nextstate(env_state, Q, visio)) \
                        + 0.15 * gamma * sumQ_nextstate(s, Q, visio)
                    if visio == True:
                        print("q=%.2f=0.8*(%.2f+%.2f*%.2f)+0.15*(%.2f*%.2f)+0.05*(%.2f+%.2f*%.2f)" % (q,rewards, gamma,
                                                                                                      sumQ_nextstate(next_state, Q,visio), gamma,
                                                                                                      sumQ_nextstate(s,Q,visio),
                                                                                                      env_rewards,gamma,
                                                                                                      sumQ_nextstate(env_state, Q,visio)))
                delta = max(delta, np.abs(q - Q.loc[s, a]))  # 更新差值
                Q.loc[s, a] = q  # 存储(更新)每个状态下的状态值函数，即伪代码中的 v <- V(s)
        iter += 1
        print('k=', iter)  # 打印迭代次数
        k = 0
        Q1 = pd.DataFrame(
            np.zeros((length, len(action))),  # q_table initial values
            columns=action,  # actions's name
        )
        for s in suqe:
            Q1.loc[k] = Q.loc[s]
            k = k + 1
        Q1.rename(columns={'n': 'UP', 's': 'DOWN', 'w': 'LEFT', 'e': 'RIGHT'}, inplace=True)
        print(Q1)
        if delta < theta:  # 策略评估的迭代次数不能太多，否则状态值函数的数值会越来越大（即使算法仍然在收敛）
            break
        # CaValue(Q)
    return Q


initPolicy()
q = policy_eval_Q_random()
CaValue(q)