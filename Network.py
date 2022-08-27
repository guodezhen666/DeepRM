import torch
import torch.nn as nn
import parameters

class Net(nn.Module):
    # n_layer: the number of hidden layers
    # n_hidden: the number of vertices in each layer
    # def __init__(self, n_layer, n_hidden, input_height, input_width, output_length):
    #     super(Net, self).__init__()
    #     self.input_length, self.output_length = input_height*input_width, output_length
    #     self.input_layer = nn.Linear(self.input_length, n_hidden)
    #     self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) if i%2==0 else nn.BatchNorm1d(n_hidden) for i in range(n_layer)])
    #     # self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(n_layer)])
    #     self.output_layer = nn.Linear(n_hidden, self.output_length)
        
    # def forward(self, x):
    #     o = self.act(self.input_layer(x))
    #     for i, li in enumerate(self.hidden_layers):
    #         o = self.act(li(o))
    #     out = nn.functional.softmax(self.output_layer(o)) # 
    #     return out

    # def act(self, x):
    #     # return x * torch.sigmoid(x)
    #     # return torch.sigmoid(x)
    #     # return torch.tanh(x)
    #     return torch.relu(x)

    # Covolutional Neural Network
    def __init__(self, n_layer, n_hidden, input_height, input_width, output_length):
        super().__init__()
        # input channels output channels kernels_size
        self.conv1 = nn.Conv2d(1, 6, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 2 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_length)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x),1)
        return x