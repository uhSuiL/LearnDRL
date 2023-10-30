# 深度强化学习——原理、算法与PyTorch实战，代码名称：代02-搭建扫地机器人的Gym环境.py

import gym
from gym import spaces
from gym.utils import seeding
import sys
from time import sleep
import signal


class Grid(object):
    def __init__(
            self,
            x: int = None,  # 坐标x
            y: int = None,  # 坐标y
            grid_type: int = 0,  # 类别值（0：空；1：障碍或边界）
            enter_reward: float = 0.0):  # 进入该格子的即时奖励
        self.x = x
        self.y = y
        self.grid_type = grid_type
        self.enter_reward = enter_reward
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "Grid: {name:{3}, x:{0}, y:{1}, grid_type:{2}}".format(self.x, self.y, self.grid_type, self.name)


class GridMatrix(object):
    def __init__(
            self,
            n_width: int,  # 水平方向格子数
            n_height: int,  # 竖直方向格子数
            default_type: int = 0,  # 默认类型，0-空
            default_reward: float = 0.0,  # 默认即时奖励值
    ):
        self.n_height = n_height
        self.n_width = n_width
        self.default_reward = default_reward
        self.default_type = default_type
        self.grids = None  # list(Grid) 将二维的格子世界中的格子存储在一维的列表中
        self.len = n_width * n_height  # 格子数
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward))

    def get_grid(self, x, y=None):
        """
        获取一个格子信息
        args: 坐标信息，由x，y表示或仅有一个类型为tuple的x表示
        return: grid object
        """
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert (0 <= xx < self.n_width and 0 <= yy < self.n_height)  # 任意坐标值应在合理区间
        index = yy * self.n_width + xx  # 二维坐标展开为一维线性坐标
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.enter_reward = reward
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, grid_type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.grid_type = grid_type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.enter_reward

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.grid_type


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(
            self,
            n_width: int = 5,  # 格子世界宽度（以格子数计）
            n_height: int = 5,  # 高度
            u_size=40,  # 当前格子绘制尺寸
            default_reward: float = 0,
            default_type=0):
        self.n_width = n_width
        self.n_height = n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self.u_size = u_size
        self.screen_width = u_size * n_width  # 场景宽度
        self.screen_height = u_size * n_height  # 场景长度

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type)
        self.reward = 0  # for rendering
        self.action = None  # for rendering

        # 0,1,2,3 represent left, right, up, down
        self.action_space = spaces.Discrete(4)
        # 观察空间由low和high决定
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)

        self.state = None  # 格子世界的当前状态
        self.ends = [(0, 0), (4, 3)]  # 终止格子坐标，可以有多个
        self.start = (0, 4)  # 起始格子坐标，只有一个
        self.types = [(2, 2, 1)]
        self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]
        self.refresh_setting()
        self.viewer = None  # 图形接口对象
        self.seed()  # 产生一个随机子
        self.reset()

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.action = action  # action for rendering
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        if action == 0:
            new_x -= 1  # left
        elif action == 1:
            new_x += 1  # right
        elif action == 2:
            new_y += 1  # up
        elif action == 3:
            new_y -= 1  # down
        elif action == 4:
            new_x, new_y = new_x - 1, new_y - 1
        elif action == 5:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 6:
            new_x, new_y = new_x + 1, new_y - 1
        elif action == 7:
            new_x, new_y = new_x + 1, new_y + 1
        # boundary effect
        if new_x < 0:
            new_x = 0
        if new_x >= self.n_width:
            new_x = self.n_width - 1
        if new_y < 0:
            new_y = 0
        if new_y >= self.n_height:
            new_y = self.n_height - 1

        # wall effect:
        # 类型为1的格子为障碍格子，不可进入
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)
        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        # 提供格子世界所有的信息在info内
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        return self.state, self.reward, done, info

    # 将状态变为横纵坐标
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # 未知状态

    def refresh_setting(self):
        '''用户在使用该类创建格子世界后可能会修改格子世界某些格子类型或奖励值
        的设置，修改设置后通过调用该方法使得设置生效。
        '''
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    # 判断是否是终止状态
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), "坐标数据不完整"
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    # 图形化界面
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0, 0)
        u_size = self.u_size
        m = 2  # 格子之间的间隙尺寸

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # 绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x * u_size + m, y * u_size + m), ((x + 1) * u_size - m, y * u_size + m),
                         ((x + 1) * u_size - m, (y + 1) * u_size - m), (x * u_size + m, (y + 1) * u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    r = self.grids.get_reward(x, y) / 10
                    if r < 0:
                        rect.set_color(0.9 - r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    # 绘制边框
                    v_outline = [(x * u_size + m, y * u_size + m), ((x + 1) * u_size - m, y * u_size + m),
                                 ((x + 1) * u_size - m, (y + 1) * u_size - m), (x * u_size + m, (y + 1) * u_size - m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)

                    if self._is_end_state(x, y):
                        # 给终点方格添加金黄色边框
                        outline.set_color(0.9, 0.9, 0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x, y) == 1:  # 障碍格子用深灰色表示
                        rect.set_color(0.3, 0.3, 0.3)
                    else:
                        pass
            # 绘制个体
            self.agent = rendering.make_circle(u_size / 4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            # 更新个体位置
        x, y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x + 0.5) * u_size, (y + 0.5) * u_size)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def CtrlCHandler(signum, frame):
    env.close()
    print("User interrupt!")
    sys.exit(0)


if __name__ == "__main__":
    env = GridWorldEnv()  # 格子世界环境类动作由0：左，1：右，2：上，3:下
    env.refresh_setting()
    env.seed(1)  # 提供随机种子
    print("env.action_space:", env.action_space)
    print("env.observation_space:", env.observation_space)
    signal.signal(signal.SIGINT, CtrlCHandler)  # 用户中断信号处理
    episode_num = 100
    for e in range(episode_num):
        env.reset()
        while True:
            action = env.action_space.sample()
            env.render()
            sleep(0.5)
            _, _, done, _ = env.step(action)
            if done:
                break
    # input("Press ctrl-c to interrupt...")
    env.close()
    sys.exit(0)
