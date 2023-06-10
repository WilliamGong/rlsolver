import gym
from gym import spaces
import pandas as pd
import numpy as np
import sys
import random
import time

randseed = int(time.time())
random.seed(randseed)  

class BusEnv(gym.Env):
    def __init__(self, 
                 data_path, 
                 c_f=13.32, c_v=2.41, 
                 max_car_num=30, 
                 h=2, delta=1, 
                 avg_time_cost=0.96, 
                 aep=0.5, 
                 car_capacity=6, 
                 max_people=500, 
                 max_wait_time=10, 
                 m=sys.maxsize):

        super().__init__()

        self.c_f = c_f
        self.c_v = c_v
        self.h = h
        self.delta = delta
        self.avg_time_cost = avg_time_cost
        self.aep = aep
        self.car_capacity = car_capacity
        self.max_people = max_people
        self.max_wait_time = max_wait_time
        self.m = m
        self.arrival_start = 0
        self.max_car_num = max_car_num

        df = pd.read_excel(data_path)
        #df = df.drop(columns='expected arrival')
        self.slot_num = df['time'].count()
        self.arrival_people = df['number'].to_list()

        state_space_shape = []
        action_space_shape = []
        for i in range(self.slot_num+1):
            state_space_shape.append(max_car_num)
            action_space_shape.append(2)

        # init state space
        self.observation_space = spaces.MultiDiscrete(state_space_shape)
        self.action_space = spaces.MultiDiscrete(action_space_shape)

        self.state = []
        for i in range(self.slot_num+1):
            #self.state.append(max_car_num / 2)
            self.state.append(random.randint(0, max_car_num))

    def reset(self):
        for i in range(self.slot_num+1):
            #self.state[i] = self.max_car_num / 2
            self.state[i] = random.randint(0, self.max_car_num)

        return np.array(self.state)

    def step(self, action):

        done = False
        reward = 0

        for i in range(len(action)):
            if action[i] == 1:
                new_val = self.state[i] - 1
                if new_val < 0:
                    done = True
                    reward -= 1e14
                else:
                    self.state[i] = new_val
            elif action[i] == 2:
                new_val = self.state[i] + 1
                if new_val > self.max_car_num:
                    done = True
                    reward -= 1e14
                else:
                    self.state[i] = new_val

        reward += self.getReward()

        return np.array(self.state), reward, done, {}

    def getSystemPeople(self):
        system_people = []
        system_people.append(0)

        for i in range(1, self.slot_num+1):
            system_people.append(max((system_people[i-1] + self.arrival_people[i-1] - self.car_capacity*self.state[i]), 0)) # arrival_people 中序号为0-n-1，与 j 错了一位

        return system_people

    def getCarcost(self, carCount):
        if carCount == 0:
            return 0
        return carCount * self.car_capacity * self.c_v * self.aep + self.c_f

    def getDeparture(self):
        isDeparture = []
        for i in self.state:
            if i > 0:
                isDeparture.append(1)
            else:
                isDeparture.append(0)
        return isDeparture

    def getObj(self):

        system_people = self.getSystemPeople()

        obj1 = sum(self.getCarcost(self.state[i]) for i in range(self.slot_num+1))
        obj2 = sum(self.delta*self.avg_time_cost*system_people[j-1] + self.delta*self.avg_time_cost*self.state[j-1]/2 for j in range(1,self.slot_num+1))
        obj = obj1 + obj2 + self.delta * self.avg_time_cost * self.arrival_start / 2

        return obj

    def getReward(self):

        system_people = self.getSystemPeople()
        obj = self.getObj()

        # s.t.
        st2 = 0
        isDeparture = self.getDeparture()
        for j in range(int(self.h/self.delta), self.slot_num+1):
            st2 += max(0, (sum(isDeparture[l] for l in range(j-int(self.h/self.delta)+1,j+1)) - 1))**2
        st5 = (system_people[self.slot_num])**2
        st6 = 0
        #每一个时隙的等车人数都不能大于Q，否则会造成乘客等待时间大于规定值。
        for j in range(self.slot_num+1):
            st6 += max(0, (system_people[j] - self.max_people))**2
        st7 = 0
        for j in range(int(self.max_wait_time/self.delta), self.slot_num+1):
            st7 += max(0, -1*(sum(self.state[l] for l in range(j-int(self.max_wait_time/self.delta)+1,j+1))-1))**2

        reward = -1 * obj - (st2 + st5 + st6 + st7)
        return reward

    def print(self):
        print('objective: {}'.format(self.getObj()))

    def getState(self):
        return self.state