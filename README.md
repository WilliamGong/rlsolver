# RL Slover
基于特定动态规划问题的强化学习求解器    
使用 DDPG（自写），PPO（stable_baselines3）进行强化学习
## 依赖
+ gym
+ numpy
+ pandas
+ openpyxl
+ stable_baselines3（用于 PPO）
+ pytorch

## 使用
使用 PPO：

    python ppo-baselines.py

使用 DDPG：

    python ddpg_v3.py

## 说明
实际测试中发现 DDPG 效果较差    
DDPG 和 PPO 中的参数均不是最优