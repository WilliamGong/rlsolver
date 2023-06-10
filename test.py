import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

from rl_env_v3 import BusEnv

box = spaces.Box(0, 300, shape=(1,), dtype=np.int64)
discrete = spaces.Discrete(300)

print(discrete.sample())

env = BusEnv('data/test-1h.xls')
print(env.observation_space)
check_env(env)
print(discrete.shape)
