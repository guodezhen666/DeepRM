import numpy as np
from Network import Net
from train import Train
import torch
from parameters import Parameters
import environment


class DataSet():
    def __init__(self, input_height, input_width, output_length, n_sample):
        self.input_length, self.output_length = input_height*input_width, output_length
        self.n_sample = n_sample
    
    def random_data_set(self):
        self.input_data = torch.randint(0, 2, (self.n_sample, self.input_length), dtype=torch.float)
        self.output_data = torch.rand([self.n_sample, self.output_length])
        return self.input_data, self.output_data
    
    def get_random_action_data(self,pa):
        env = environment.Env(pa,render=False, repre='image')
        action = np.random.randint(0,pa.network_output_dim,pa.simu_len*pa.num_ex)
        observation = np.zeros((pa.simu_len*pa.num_ex,pa.network_input_height*pa.network_input_width),dtype=np.float32)
        rewards = np.zeros((pa.simu_len*pa.num_ex,1),dtype=np.float32)
        for j in range(pa.num_ex):
            env.reset()
            for i in range(pa.simu_len):
                ob, reward, done, info = env.step(action[i*j])
                observation[i*j,:] = ob.reshape(-1)
                rewards[i*j,:] = reward
            
        for j in range(pa.num_ex): # number of examples
            for i in range(pa.simu_len-1,0,-1):
                p = i*(j+1)
                rewards[p-1,:] = rewards[p-1,:] + pa.discount*rewards[p,:] # cumulate reward
            
        observation = torch.from_numpy(observation)
        rewards = torch.from_numpy(rewards)
        return observation,rewards
    
    
    def get_shortest_job_first_data(self,pa):
        env = environment.Env(pa,render=False, repre='image')
        action = np.zeros(pa.simu_len*pa.num_ex,dtype=np.int8)
        observation = np.zeros((pa.simu_len*pa.num_ex,pa.network_input_height*pa.network_input_width),dtype=np.float32)
        rewards = np.zeros((pa.simu_len*pa.num_ex,1),dtype=np.float32)
        for j in range(pa.num_ex):
            env.reset()
            for i in range(pa.simu_len):
                action[i*j] = get_sjf_action(env.machine, env.job_slot)
                ob, rew, done, info = env.step(action[i*j], repeat=True)
                if done:  # hit void action, exit
                    break
        
        # get observation and reward
        for j in range(pa.num_ex):
            env.reset()
            for i in range(pa.simu_len):
                ob, reward, done, info = env.step(action[i*j])
                observation[i*j,:] = ob.reshape(-1)
                rewards[i*j,:] = reward
        
        # accumulative reward
        for j in range(pa.num_ex): # number of examples
            for i in range(pa.simu_len-1,0,-1):
                p = i*(j+1)
                rewards[p-1,:] = rewards[p-1,:] + pa.discount*rewards[p,:] # cumulate reward
            
        observation = torch.from_numpy(observation)
        rewards = torch.from_numpy(rewards)
        return observation,rewards

          
def get_sjf_action(machine, job_slot):
    sjf_score = 0
    act = len(job_slot.slot)  # if no action available, hold

    for i in range(len(job_slot.slot)):
        new_job = job_slot.slot[i]
        if new_job is not None:  # there is a pending job

            avbl_res = machine.avbl_slot[:new_job.len, :]
            res_left = avbl_res - new_job.res_vec

            if np.all(res_left[:] >= 0):  # enough resource to allocate

                tmp_sjf_score = 1 / float(new_job.len)

                if tmp_sjf_score > sjf_score:
                    sjf_score = tmp_sjf_score
                    act = i
    return act        

if __name__ == '__main__':
    pa = Parameters()
    input_height, input_width = pa.network_input_height, pa.network_input_width
    output_length = pa.network_output_dim
    n_sample = pa.simu_len
    data_set = DataSet(input_height, input_width, output_length, n_sample)
    net = Net(n_layer = 4, n_hidden = 10, input_height=input_height, input_width=input_width, output_length=output_length)
    train = Train(net = net, data_set = data_set, BATCH_SIZE = pa.batch_size, pa = pa)
    train.train(epoch = pa.num_epochs, lr = pa.lr_rate)
    train.save_model()
    
