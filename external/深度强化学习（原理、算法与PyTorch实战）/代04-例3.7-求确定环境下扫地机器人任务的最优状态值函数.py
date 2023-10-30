'''
深度强化学习——原理、算法与PyTorch实战
'''
import numpy as np

class SweepRobot():
    def __init__(self):
        # 状态空间
        self.S = [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
                  [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]],
                  [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5]],
                  [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 5]],
                  [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
                  [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]]]
        # 动作空间
        self.A = [[None, None], [-1, 0], [1, 0], [0, -1], [0, 1]]
        # 状态值
        self.V = [[None for i in range(6)] for j in range(6)]
        self.V[1][1] = 0
        self.V[5][4] = 0
        # 无策略
        self.gamma = 0.8
        self.theta = 0.0001

    def reward(self, s, a):
        # 奖励函数
        [truth1, truth2] = np.add(s, a) == [5, 4]
        [truth3, truth4] = np.add(s, a) == [1, 1]
        [truth5, truth6] = np.add(s, a) == [3, 3]
        # 若状态s转移到[5,4](收集垃圾)
        if s != [5, 4] and (truth1 and truth2):
            return 3
        # 若状态s转移到[1,1](充电)
        if s != [1, 1] and (truth3 and truth4):
            return 1
        # 若状态s转移到[3,3](撞到障碍物)
        if truth5 and truth6:
            return -10
        return 0

    def cal_optimal_value(self):
        # 建立V的副本
        copy_V = self.V
        # 首先初始化V值,便于计算，都初始化为0
        for i in range(1, 6):
            for j in range(1, 6):
                # 判断是否是终止情况，如果是的话直接计算下一个
                [truth1, truth2] = [i == 5, j == 4]
                [truth3, truth4] = [i == 1, j == 1]
                [truth5, truth6] = [i == 3, j == 3]
                if truth1 and truth2:
                    continue
                if truth3 and truth4:
                    continue
                if truth5 and truth6:
                    continue
                self.V[i][j] = 0
                copy_V[i][j] = 0
        while True:
            Delta = 0
            for i in range(1, 6):
                for j in range(1, 6):
                    # 判断是否是终止情况，如果是的话直接计算下一个
                    [truth1, truth2] = [i == 5, j == 4]
                    [truth3, truth4] = [i == 1, j == 1]
                    [truth5, truth6] = [i == 3, j == 3]
                    if truth1 and truth2:
                        continue
                    if truth3 and truth4:
                        continue
                    if truth5 and truth6:
                        continue
                    v = self.V[i][j]
                    # 因为每个状态的动作空间不一样，所以需要分情况讨论
                    max_value = 0
                    # 向上的情况
                    if i - 1 >= 1:
                        if i - 1 == 3 and j == 3:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[1]) + self.gamma * self.V[i][j])
                        else:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[1]) + self.gamma * self.V[i - 1][j])
                    # 向下的情况
                    if i + 1 <= 5:
                        if i + 1 == 3 and j == 3:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[1]) + self.gamma * self.V[i][j])
                        else:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[2]) + self.gamma * self.V[i + 1][j])
                    # 向左的情况
                    if j - 1 >= 1:
                        if j - 1 == 3 and i == 3:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[1]) + self.gamma * self.V[i][j])
                        else:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[3]) + self.gamma * self.V[i][j - 1])
                    # 向右的情况
                    if j + 1 <= 5:
                        if j + 1 == 3 and i == 3:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[1]) + self.gamma * self.V[i][j])
                        else:
                            max_value = max(max_value, self.reward(self.S[i][j], self.A[4]) + self.gamma * self.V[i][j + 1])
                    copy_V[i][j] = max_value
                    Delta = max(Delta, abs(v - copy_V[i][j]))
            self.V = copy_V
            if Delta < self.theta:
                break

    def print_value(self):
        # 输出扫地机器人的状态值
        print('扫地机器人最优状态值：')
        for i in range(1, 6):
            for j in range(1, 6):
                if self.V[j][6 - i] != None:
                    print('%.3f'%self.V[j][6 - i], end=" ")
                else:
                    print(self.V[j][6 - i], end=" ")
            print()

if __name__ == '__main__':
    sr = SweepRobot()
    sr.cal_optimal_value()
    sr.print_value()