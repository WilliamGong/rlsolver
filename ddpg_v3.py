from rl_env_v3 import BusEnv

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter 

# choose cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Net
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor,self).__init__()
        self.action_bound = torch.tensor(action_bound)
        n1_layer = 64
        n2_layer = 32
        # layer
        self.layer_1 = nn.Linear(state_dim, n1_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.3) # 初始化权重
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(n1_layer, n2_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.3) # 初始化权重
        nn.init.constant_(self.layer_2.bias, 0.1)
        
        self.output = nn.Linear(n2_layer, action_dim)
        self.output.weight.data.normal_(0.,0.3) # 初始化权重
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.relu(self.layer_2(a))
        a = torch.tanh(self.output(a))
        # 对action进行放缩，实际上a in [-1,1]
        scaled_a = torch.round((a + 1.0) * self.action_bound/2)
        return scaled_a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()
        n1_layer = 64
        n2_layer = 32
        # layer
        self.layer_s1 = nn.Linear(state_dim, n1_layer)
        nn.init.normal_(self.layer_s1.weight, 0., 0.1)
        nn.init.constant_(self.layer_s1.bias, 0.1)

        self.layer_s2 = nn.Linear(n1_layer, n2_layer)
        nn.init.normal_(self.layer_s2.weight, 0., 0.3) # 初始化权重
        nn.init.constant_(self.layer_s2.bias, 0.1)
        
        self.layer_a1 = nn.Linear(action_dim, n1_layer)
        nn.init.normal_(self.layer_a1.weight, 0., 0.1)
        nn.init.constant_(self.layer_a1.bias, 0.1)

        self.layer_a2 = nn.Linear(n1_layer, n2_layer)
        nn.init.normal_(self.layer_a2.weight, 0., 0.3) # 初始化权重
        nn.init.constant_(self.layer_a2.bias, 0.1)
        
        self.output = nn.Linear(n2_layer, 1)

    def forward(self,s,a):
        s = self.layer_s1(s)
        s = self.layer_s2(s)
        a = self.layer_a1(a)
        a = self.layer_a2(a)
        q_val = self.output(torch.relu(s+a))
        return q_val

# DDPG
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement,memory_capacity=1000,gamma=0.9,lr_a=0.001, lr_c=0.002,batch_size=32) :
        super(DDPG,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacticy = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # 初始化 Actor 网络
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        # 初始化 Critic 网络
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()
    
	# 从记忆库中随机采样    
    def sample(self):
        indices = np.random.choice(self.memory_capacticy, size=self.batch_size)
        return self.memory[indices, :] 

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(device)
        action = self.actor(s)
        return action.detach().cpu().numpy()

    def learn(self):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0]+'.weight']
                al[1].weight.data.mul_((1-tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0]+'.weight'])
                al[1].bias.data.mul_((1-tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0]+'.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1-tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0]+'.weight'])
                cl[1].bias.data.mul_((1-tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0]+'.bias'])
            
        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0]+'.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0]+'.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0]+'.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0]+'.bias']
            
            self.t_replace_counter += 1

        # 从记忆库中采样 batch data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim]).to(device)
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim]).to(device)
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim]).to(device)
        bs_ = torch.FloatTensor(bm[:,-self.state_dim:]).to(device)
        
        # 训练Actor
        a = self.actor(bs.to(device))
        q = self.critic(bs.to(device), a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()
        
        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target,q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

        return a_loss, td_error

    # 存储序列数据
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.memory_capacticy
        self.memory[index, :] = transition
        self.pointer += 1



if __name__ == '__main__':
    # Define some Hyper Parameters
    BATCH_SIZE = 64     # batch size of sampling process from buffer
    LR_A = 0.0001           # learning rate
    LR_C = 0.0001
    EPSILON = 0.9       # epsilon used for epsilon greedy approach
    GAMMA = 0.9         # discount factor
    TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
    MEMORY_CAPACITY = 200                  # The capacity of experience replay buffer
    MAX_EPISODES = 1000
    #MAX_EP_STEPS = 300
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]
    VAR = 5  # control exploration

    file_path = 'data/test-1h.xls'

    # Tensorboard
    tensorboard_path = './tf_dir'
    writer = SummaryWriter(tensorboard_path)

    # output
    cars = []

    env = BusEnv(file_path)
    env = env.unwrapped
    action_dim = 1  
    state_dim = env.observation_space.shape[0]
    action_bound = 30
    ddpg = DDPG(state_dim=state_dim,
                action_dim=action_dim,
                action_bound=action_bound,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY, 
                gamma=GAMMA, 
                lr_a=LR_A, lr_c=LR_C, 
                batch_size=BATCH_SIZE)
                
    t1 = time.time()
    losses_counter = 0
    for i in range(MAX_EPISODES):

        s = env.reset()
        ep_reward = 0
        cars = []

        while True:

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.round(np.clip(np.random.normal(a, VAR), 0, action_bound))  # 在动作选择上添加随机噪声
            cars.append(a[0])

            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .9995  # decay the action randomness
                a_loss, td_err = ddpg.learn()
                losses_counter += 1

                writer.add_scalar('loss/a_loss', a_loss, i)
                writer.add_scalar('loss/td_err', td_err, i)
                

            s = s_
            if done:
                obj = env.getObj()
                ep_reward = env.getReward()

                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                print('objective: {}'.format(obj))
                if env.getIsDone():
                    print("Done")
                print(env.true_actions, len(env.true_actions))

                writer.add_scalar('objective', obj, i)
                writer.add_scalar('reward', ep_reward, i)
                break

    print('Running time: ', time.time() - t1)

    # output
    print('Output')
    df = pd.read_excel(file_path)
    time = df['time'].to_list()
    #cars = env.getState()
    table = list(zip(time, env.true_actions))
    df_output = pd.DataFrame(table, columns=['time', 'cars'])
    df_output.to_excel('output.xlsx')