# 代13-例4.8-基于动作值函数的确定环境扫地机器人任务值迭代
import numpy as np
import pandas as pd

"""定义格子世界参数"""
world_h = 5
world_w = 5
length = world_h * world_w
gamma = 0.8
state = [i for i in range(length)]  # 状态（编号）
action = ['n', 's', 'w', 'e']  # 动作名称
ds_action = {'n': -world_w, 'e': 1, 's': world_w, 'w': -1}
value = [0 for i in range(length)]  # 初始化状态值函数，均为0.  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        return 2
    elif a == 'e':
        return 1
    elif a == 's':
        return 3
    elif a == 'w':
        return 0


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


# 更新状态值函数
def value_update(s, numb):  # 传入当前状态
    value_new = 0
    if s in [9, 20, 12]:  # 若当前状态为吸入状态，则直接pass不做操作
        pass
    else:
        Q = []
        successor = getsuccessor(s)  # 得到所有可能的下一状态list
        # rewards = reward(s)  # 得到当前状态的奖励
        # print("s %s="%s,end="")
        for next_state in successor:  # 遍历所有可能的下一状态
            rewards = reward(next_state)
            value_new = rewards + gamma * value[next_state]  # 计算公式，得到当前状态的状态价值函数
            Q.append(value_new)
            # print("%.2f*(%d+%.1f*%.2f) = %.2f"%(1/len(successor),rewards,gamma,value[next_state],value_new))
            # 注意前面的1/len(successor)为该s状态能够到下个状态的个数概率，该代码是第一次迭代时的固定策略π(a|s)
        value_new = max(Q)
        # print("第%d个状态最大值：%.2f"%(numb,value_new))
        # print()
    return value_new


def Caculate_Q(s, V, num, Q, discount_factor=0.8):
    """
    Returns:
        A vector of length env.nA containing the expected value of each action.
    """

    #     Q = np.zeros((length, 4))
    # Q[:][:] = -1000
    for a in action:  # 遍历能走的动作
        # for prob, next_state, reward, done in env.P[s][a]:
        #     Q[s][a] += prob * (reward + discount_factor * V[next_state])  # 计算当前状态s下的动作值函数列表 [q1,q2,q3,q4]
        next_state = next_states(s, a)
        if next_state == s:  # 碰壁
            continue
        rewards = reward(next_state)
        numberA = getAction(a)
        Q.loc[s, a] = rewards + discount_factor * V[next_state]


#         print("Q[%s][%d]  %.2f = %s + 0.8 * %.2f:"%(num,numberA,Q[s][numberA],rewards,V[next_state]))


def initial_state():
    v = np.array(value).reshape(world_h, world_w)  # 调整初始化状态值函数矩阵
    print(v)


def main():
    max_iter = 7
    initial_state()

    iter = 1
    Q = pd.DataFrame(
        np.zeros((length, len(action))),  # q_table initial values
        columns=action,  # actions's name
    )
    while iter < max_iter:
        numb = -1
        for s in suqe:  # 遍历所有状态
            if s in [9, 20, 12]:  # 若当前状态为吸入状态，则直接pass不做操作
                continue
            numb += 1
            value[s] = value_update(s, numb)  # 更新状态值函数
            Caculate_Q(s, value, numb, Q)
        Q1 = pd.DataFrame(
            np.zeros((length, len(action))),  # q_table initial values
            columns=action,  # actions's name
        )
        k = 0
        for s in suqe:
            Q1.loc[k] = Q.loc[s]
            k = k + 1
        Q1.rename(columns={'n': 'UP', 's': 'DOWN', 'w': 'LEFT', 'e': 'RIGHT'}, inplace=True)

        v = np.array(value).reshape(world_h, world_w)  # 更新状态值函数矩阵

        if (iter <= 10) or (iter % 10 == 0):  # 前1次 + 每10次打印一次
            print('k=', iter)  # 打印迭代次数
            print(Q1)
            print(np.round(v, decimals=4))  # np.round() 返回浮点数的四舍五入值

        iter += 1


if __name__ == '__main__':
    main()
