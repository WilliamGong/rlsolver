import gym
from gym import spaces
import pandas as pd
import numpy as np
import sys

class BusEnv(gym.Env):
    def __init__(self, 
                 data_path, 
                 c_f=13.32, c_v=2.41, 
                 max_car_num=30, 
                 min_departure_time=2, delta=1, 
                 avg_time_cost=0.96, 
                 aep=0.5, 
                 car_capacity=6, 
                 max_people=500, 
                 max_wait_time=10, 
                 m=999999999):


        super().__init__()

        self.c_f = c_f
        self.c_v = c_v
        self.min_departure_time = min_departure_time
        self.delta = delta
        self.avg_time_cost = avg_time_cost
        self.aep = aep
        self.car_capacity = car_capacity
        self.max_people = max_people
        self.max_wait_time = max_wait_time
        self.m = 0
        self.arrival_start = 0
        self.max_car_num = max_car_num

        self.final_reward = 1e5
        self.push_punishment = self.final_reward * 50

        df = pd.read_excel(data_path)
        #df = df.drop(columns='expected arrival')
        self.slot_num = df['time'].count()
        self.arrival_people = df['number'].to_list()

        self.step_reward = self.push_punishment / len(self.arrival_people)

        # state
        self.system_people = 0
        self.current_time = 0
        self.is_departure = 0
        self.observation_space = spaces.MultiDiscrete([max_people, len(self.arrival_people), max_people, 2, max_car_num+1])

        # action
        self.action_space = spaces.Discrete(max_car_num+1)
        self.true_actions = []

        # init reward
        self.reward = self.m

        # init other params
        self.timer = 0
        self.done = False

    def reset(self):
        # state
        self.system_people = 0
        self.current_time = 0
        self.is_departure = 0

        # action
        self.true_actions = []

        #others
        self.reward = self.m
        self.timer = 0
        self.done = False

        return np.array([self.system_people, self.current_time, self.arrival_people[self.current_time], self.is_departure, 0])

    def step(self, action):

        if type(action) is np.ndarray and action.shape != ():
            action = action[0]

        self.done = False

        # set timer
        self.timer += 1

        # pay wait cost before arrival
        self.reward -= self.delta * self.avg_time_cost * self.system_people

        # add arrival people and pay average wait cost
        self.system_people += self.arrival_people[self.current_time]
        self.reward -= self.delta * self.avg_time_cost * self.arrival_people[self.current_time] / 2

        # action block
        if action > 0 and self.is_departure == 1 and self.current_time != len(self.arrival_people)-1:
            action = 0

        # pay car cost
        self.reward -= action * self.car_capacity * self.aep * self.c_v + self.c_f

        # departure
        self.system_people -= action * self.car_capacity
        if self.system_people < 0:
            self.system_people = 0

        # make sure that there is no person in final time slot
        if self.current_time == len(self.arrival_people)-1:
            if self.system_people == 0:
                self.reward += self.final_reward
            else:
                self.reward -= self.final_reward

            self.done = True

        # test terminate condition
        if self.system_people > self.max_people:
            self.done = True

        if self.timer > self.max_wait_time and action <= 0:
            self.done = True

        # push agent to iterate more time
        if self.current_time != len(self.arrival_people)-1 and self.done:
            self.reward -= self.push_punishment
            self.reward += self.current_time * self.step_reward

        # departure flag to control min departure spacing
        if action > 0:
            self.is_departure = 1
            self.timer = 0

        if self.timer >= np.ceil(self.min_departure_time / self.delta):
            self.is_departure = 0

        # build return state
        return_state = np.array([self.system_people, self.current_time, self.arrival_people[self.current_time], self.is_departure, action])

        # update time
        self.current_time += 1

        self.true_actions.append(int(action))
        # all done, return 
        return return_state, self.reward, self.done, {}

    def getObj(self):

        current_time = self.current_time - 1

        obj = self.m - self.reward

        if current_time == len(self.arrival_people)-1:
            if self.system_people == 0:
                obj += self.final_reward
            else:
                obj -= self.final_reward


        if current_time != len(self.arrival_people)-1 and self.done:
            obj -= self.push_punishment
            obj += (self.current_time - 1) * self.step_reward

        return obj

    def getReward(self):
        return self.reward

    def getIsDone(self):
        if self.current_time-1 == len(self.arrival_people)-1:
            return True
        else:
            return False