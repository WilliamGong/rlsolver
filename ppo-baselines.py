from rl_env_v3 import BusEnv
from stable_baselines3 import PPO 
import argparse
import pandas as pd

# set param via args
parser = argparse.ArgumentParser(description="hypertuning")
parser.add_argument("--batch_size", help="batch size, val is the n in 2^n, [4, 14]", type=int, default=6)
parser.add_argument("--lr", help="learning rate of actor, [0.00001, 0.01]", type=float, default=0.0001)
parser.add_argument("--n_steps", help="The number of steps to run for each environment per update, val is the n in 2^n, [5, 12]", type=int, default=11)
parser.add_argument("--gamma", help="discount factor, [0.9, 0.999]", type=float, default=0.99)
parser.add_argument("--clip_range", help="Clipping parameter, it can be a function of the current progress remaining, [0.1, 0.3]", type=float, default=0.2)
parser.add_argument("--n_epochs", help=" Number of epoch when optimizing the surrogate loss, [2, 20]", type=int, default=10)
parser.add_argument("--gae_lambda", help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator, [0.96, 0.99]", type=float, default=0.99)
parser.add_argument("--ent_coef", help="Entropy coefficient for the loss calculation, [0.005, 0.05]", type=float, default=0.01)
parser.add_argument("--timesteps", help="he total number of samples (env steps) to train on, [10000, 50000]", type=int, default=20000)

args = parser.parse_args()

file_path = 'data/3.25_T1.xlsx'

env = BusEnv(file_path)

model = PPO("MlpPolicy", 
            env, 
            learning_rate=args.lr, 
            n_steps=2**args.n_steps, 
            batch_size=2**args.batch_size, 
            n_epochs=args.n_epochs, 
            gamma=args.gamma, 
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1, 
           tensorboard_log='tf-dir')
model.learn(total_timesteps=args.timesteps)
model.save('ppo-busenv')

obs = env.reset()

for i in range(100):

    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #print(obs, done)

    print("reward: {}, objective: {}".format(rewards, env.getObj()))

    print(env.true_actions, len(env.true_actions))

# output
print('Output')
df = pd.read_excel(file_path)
time = df['time'].to_list()
table = list(zip(time, env.true_actions))
df_output = pd.DataFrame(table, columns=['time', 'cars'])
df_output.to_excel('output.xlsx')