'''
深度强化学习——原理、算法与PyTorch实战
'''
import numpy as np

class sweeprobot():
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
        # 策略
        self.pi = None
        self.gamma = 0.8

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

    def cal_coefficient(self):
        # 该函数用来计算出线性方程组的系数矩阵和向量值
        # 首先初始化一个25 * 25的系数矩阵和25个元素的向量
        coef_Matrix = [[0 for i in range(25)] for j in range(25)]
        b = [0 for i in range(25)]
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
                # 计算当前状态下的动作数，以用于计算策略pi
                count_action = 0
                if i - 1 >= 1:
                    count_action += 1
                if i + 1 <= 5:
                    count_action += 1
                if j - 1 >= 1:
                    count_action += 1
                if j + 1 <= 5:
                    count_action += 1
                self.pi = 1 / count_action
                # 具体计算每一个状态值的函数
                b_value = 0
                coef_CurrentState = 0
                # 向上的情况
                if i - 1 >= 1:
                    b_value = b_value + self.pi * self.reward(self.S[i][j], self.A[1])
                    if i - 1 == 3 and j == 3:
                        coef_CurrentState = self.pi * self.gamma
                    else:
                        coef1 = self.pi * self.gamma
                        coef_Matrix[(i - 1) * 5 + j - 1][((i - 1) - 1) * 5 + j - 1] = coef1
                # 向下的情况
                if i + 1 <= 5:
                    b_value = b_value + self.pi * self.reward(self.S[i][j], self.A[2])
                    if i + 1 == 3 and j == 3:
                        coef_CurrentState = self.pi * self.gamma
                    else:
                        coef2 = self.pi * self.gamma
                        coef_Matrix[(i - 1) * 5 + j - 1][((i + 1) - 1) * 5 + j - 1] = coef2
                # 向左的情况
                if j - 1 >= 1:
                    b_value = b_value + self.pi * self.reward(self.S[i][j], self.A[3])
                    if j - 1 == 3 and i == 3:
                        coef_CurrentState = self.pi * self.gamma
                    else:
                        coef3 = self.pi * self.gamma
                        coef_Matrix[(i - 1) * 5 + j - 1][(i - 1) * 5 + (j - 1) - 1] = coef3
                # 向右的情况
                if j + 1 <= 5:
                    b_value = b_value + self.pi * self.reward(self.S[i][j], self.A[4])
                    if j + 1 == 3 and i == 3:
                        coef_CurrentState = self.pi * self.gamma
                    else:
                        coef4 = self.pi * self.gamma
                        coef_Matrix[(i - 1) * 5 + j - 1][(i - 1) * 5 + (j + 1) - 1] = coef4
                # 将左边的移项，所以系数为-1 (单位矩阵减系数矩阵)
                coef_Matrix[(i - 1) * 5 + j - 1][(i - 1) * 5 + j - 1] = -1 + coef_CurrentState
                # 同理，将常数项移项需要乘-1
                b[(i - 1) * 5 + j - 1] = -1 * b_value
        # 因为状态[1,1]和状态[5,4]可以确定其状态值为0,状态[3,3]不存在，所以其实只需求22*22的矩阵和22个元素的向量值
        # 把矩阵和向量第[(1-1)*5+1-1]和[(5-1)*5+4-1]删除
        del coef_Matrix[23]
        del b[23]
        del coef_Matrix[12]
        del b[12]
        del coef_Matrix[0]
        del b[0]
        # 把矩阵每一行的[(1-1)*5+1-1]和[(5-1)*5+4-1]和[(3-1)*5+3-1]删除
        for item in coef_Matrix:
            del item[23]
            del item[12]
            del item[0]
        # 得到系数矩阵coef_Matrix = (γP-I)与 b = -R,其中γ为衰退因子，P为状态转移矩阵，I为单位矩阵，R为奖励函数
        return coef_Matrix, b

    def solve_equation(self, coef_Matrix, b):
        # 计算状态值函数
        # 解方程组A*x = b,其中A = (γP-I)，b = -R 
        A = np.array(coef_Matrix)
        b = np.array(b)
        x = np.linalg.solve(A, b)
        x = list(x)
        for i in range(1, 6):
            for j in range(1, 6):
                [truth1, truth2] = [i == 5, j == 4]
                [truth3, truth4] = [i == 1, j == 1]
                [truth5, truth6] = [i == 3, j == 3]
                if truth1 and truth2:
                    continue
                if truth3 and truth4:
                    continue
                if truth5 and truth6:
                    continue
                self.V[i][j] = x.pop(0)

    def print_value(self):
        # 输出扫地机器人的状态值
        print('扫地机器人在随机策略下的状态值：')
        for i in range(1, 6):
            for j in range(1, 6):
                if self.V[j][6 - i] != None:
                    print('%.3f'%self.V[j][6 - i], end=" ")
                else:
                    print(self.V[j][6 - i], end=" ")
            print()

if __name__ == '__main__':
    sr = sweeprobot()
    A, b = sr.cal_coefficient()
    sr.solve_equation(A, b)
    sr.print_value()