import torch
import torch.nn as nn 

class Net(nn.Module):
    # n_layer: the number of hidden layers
    # n_hidden: the number of vertices in each layer
    def __init__(self, n_layer, n_hidden, input_height, input_width, output_length):
        super(Net, self).__init__()
        self.input_length, self.output_length = input_height*input_width, output_length
        self.input_layer = nn.Linear(self.input_length, n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) if i%2==0 else nn.BatchNorm1d(n_hidden) for i in range(n_layer)])
        # self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(n_layer)])
        self.output_layer = nn.Linear(n_hidden, self.output_length)
        
    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))
        out = nn.functional.softmax(self.output_layer(o)) # 
        return out

    def act(self, x):
        # return x * torch.sigmoid(x)
        # return torch.sigmoid(x)
        # return torch.tanh(x)
        return torch.relu(x)