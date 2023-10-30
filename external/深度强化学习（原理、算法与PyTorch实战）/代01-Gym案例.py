# 深度强化学习——原理、算法与PyTorch实战，代码名称：代01-Gym案例.py

import gym  # 引入环境

env = gym.make('MountainCar-v0')  # 创建MountainCar-v0环境。
for episode in range(2):
    env.reset()  # 重置智能体状态
    print("Episode finished after {} timesteps".format(episode))
    for _ in range(500):  # 进行1000次迭代
        env.render()  # 渲染
        observation, reward, down, info = env.step(env.action_space.sample())  # 执行动作。env.action_space.sample()是随机动作选择
env.close()
