# 代07-例4.3-基于动作值函数的确定环境扫地机器人任务策略评估
import pandas as pd
import numpy as np

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


def initPolicy():
    for s in range(length):
        for a in action:
            if next_states(s, a) == s:
                continue
            newAction = getAction(a)
            policy[s][newAction] = 1 / len(getsuccessor(s))
    # print(policy)


def policy_eval(theta=0.0001):
    V = np.zeros(length)  # 初始化状态值函数列表
    iter = 0

    while True:
        k = -1
        delta = 0  # 定义最大差值，判断是否有进行更新
        for s in suqe:  # 遍历所有状态 [0~25]
            k += 1
            if s in [9, 20, 12]:  # 若当前状态为吸入状态，则直接pass不做操作
                continue
            v = 0  # 针对每个状态值函数进行计算
            # print("第%d 的状态" % (k), end="")
            for a in action:
                newAction = getAction(a)
                next_state = next_states(s, a)
                rewards = reward(next_state)
                if next_state == 12:
                    v += policy[s][newAction] * (rewards + gamma * V[s])
                    # print(" %.2f*(%d+%.1f*%.3f)+" % (policy[s][newAction], rewards, gamma, V[next_state]), end="")
                    # print(" %.2f*(%d+%.1f*%.3f)+" % (policy[s][newAction], rewards, gamma, V[next_state]), end="")
                else:
                    v += policy[s][newAction] * (rewards + gamma * V[next_state])
                    #     print("%.2f*(%d+%.1f*%.2f)+" % (policy[s][newAction], rewards, gamma, value[next_state]), end="")
                    # print()
                    # successor = getsuccessor(s)
                    # for next_state in successor:
                    #     rewards = reward(next_state)
                    #     v += 1 / len(successor) * (rewards + gamma * V[next_state])
                    # print(" %.2f*(%d+%.1f*%.3f)+" % (policy[s][newAction], rewards, gamma, V[next_state]), end="")
            # print("v = %.3f" % (v))

            delta = max(delta, np.abs(v - V[s]))  # 更新差值
            V[s] = v  # 存储(更新)每个状态下的状态值函数，即伪代码中的 v <- V(s)
        value = np.array(V).reshape(world_h, world_w)
        iter += 1
        # print('k=', iter)  # 打印迭代次数
        # print("当前的状态值函数为：")
        # print(np.round(value, decimals=3))# 输出当前的状态值函数
        if delta < theta:  # 策略评估的迭代次数不能太多，否则状态值函数的数值会越来越大（即使算法仍然在收敛）
            break
    return V  # 一轮迭代结束后，状态值函数暂时固定

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

def policy_eval_Q(theta=0.0001):
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
            for a in action:
                newAction = getAction(a)
                next_state = next_states(s, a)
                rewards = reward(next_state)
                if policy[s][newAction] == 0:
                    continue
                if next_state == 12:
                    q = rewards + gamma * (sumQ_nextstate(s, Q, visio))
                else:
                    q = rewards + gamma * (sumQ_nextstate(next_state, Q, visio))
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
value = policy_eval()
q = policy_eval_Q()
CaValue(q)



