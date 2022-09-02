# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:12:17 2022

@author: ZHEN DEGUO
"""

from environment import Env
import parameters
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import DQN
import numpy as np
from data_set import get_sjf_action
import torch
import matplotlib.pyplot as plt


# def test_stable_baselines3():
#     pa = parameters.Parameters()
#     env = Env(pa, render=False, repre='image')
#     check_env(env)
#     model = DQN('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=int(2000))
#     model.save("dqn_test")



def get_slow_down_cdf(info):
    jobs_slow_down = []
    enter_time = np.array([info[i].enter_time for i in range(len(info))])
    finish_time = np.array([info[i].finish_time for i in range(len(info))])
    job_len = np.array([info[i].len for i in range(len(info))])
    
    finished_idx = (finish_time >= 0)
    # unfinished_idx = (finish_time < 0)
    jobs_slow_down.append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )
    slow_down_cdf = np.sort(np.concatenate(jobs_slow_down))
    slow_down_yvals = np.arange(len(slow_down_cdf))/float(len(slow_down_cdf))
    
    return slow_down_cdf,slow_down_yvals
    


def compare_policy_delay():
    # random action delay
    pa = parameters.Parameters()
    pa.num_ex = 1
    env = Env(pa,render=False, repre='image',end="all_done")
    for i in range(pa.episode_max_length):
        action = np.random.randint(0,pa.num_nw)
        ob, reward, done, info_random_action = env.step(action)
        # print(len(info))
    
    # SJF action delay
    env.reset()
    for i in range(pa.episode_max_length):
        action = get_sjf_action(env.machine, env.job_slot)
        ob, reward, done, info_SJF = env.step(action)
        
        
    # reinforcement learning model
    env.reset()
    model = torch.load("net_model.pkl")
    model.eval()
    action = 0
    for i in range(pa.episode_max_length):
        ob, reward, done, info_RL = env.step(action)
        state = np.zeros((1,1,20,124))
        state[0,0,:,:] = ob
        state = torch.tensor(state,dtype=torch.float32)
        action = model(state)[0].argmax().item()
        print(action)
    
    # print(info_SJF)
    # print(info_RL)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    slow_down_cdf_random ,slow_down_yvals_random = get_slow_down_cdf(info_random_action)
    ax.plot(slow_down_cdf_random, slow_down_yvals_random, linewidth=2,label="random")
    slow_down_cdf_SJF ,slow_down_yvals_SJF = get_slow_down_cdf(info_SJF)
    ax.plot(slow_down_cdf_SJF, slow_down_yvals_SJF, linewidth=2,label="SJF")
    slow_down_cdf_RL ,slow_down_yvals_RL = get_slow_down_cdf(info_RL)
    ax.plot(slow_down_cdf_RL, slow_down_yvals_RL, linewidth=2,label="RL")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    compare_policy_delay()