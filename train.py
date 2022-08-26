from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np

class Train():
    def __init__(self, net, data_set, BATCH_SIZE, pa):
        self.errors = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.data_set = data_set
        # random action data
        # self.state, self.reward = self.data_set.get_random_action_data(pa) # self.state:(n_sample, input_length), self.reward:(n_sample, 1)
        # shortest job first action data
        self.state, self.reward = self.data_set.get_shortest_job_first_data(pa)
        
    def train(self, epoch, lr):
        optimizer = optim.Adam(self.net.parameters(), lr)
        avg_loss = 0
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.loss_func(self.BATCH_SIZE)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            # if e % 100 == 99:
            loss = avg_loss/50
            print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss))
            avg_loss = 0

            error = self.loss_func(2**8)
            self.errors.append(error.detach())

    def loss_func(self, BATCH_SIZE):
        self.x = Variable(self.state) # (n_sample, input_length)
        self.action = self.net(self.x) # (n_sample, output_length)
        self.weights = torch.zeros_like(self.action) # (n_sample, output_length)
        self.weights += self.reward # (n_sample, output_length)
        self.cost = torch.neg(torch.sum(torch.log(self.action)*self.weights)) # max->min
        return self.cost

    def get_errors(self):
        return self.errors

    def save_model(self):
        torch.save(self.net, 'net_model.pkl')
    
    def plot_kpi(self):
        fig = plt.figure()
        plt.plot(np.log(self.errors), '-b', label='Errors')
        plt.title('Training Loss', fontsize=10)
        plt.show()