import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import utils
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class Net_one_layer(nn.Module):
    def __init__(self, size_layer = 200):
        super().__init__()
        self.size_layer = size_layer
        self.fc1 = nn.Linear(1, self.size_layer)
        torch.nn.init.uniform_(self.fc1.bias,a=-1.0, b = 1.0)
        torch.nn.init.uniform_(self.fc1.weight,a=-1.0, b =2.0)

        self.fc2 = nn.Linear(self.size_layer, 1)
        torch.nn.init.normal_(self.fc2.weight,mean=2/self.size_layer, std = 0.01)
        torch.nn.init.normal_(self.fc2.bias,mean=0.0, std = 0.001)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y

class Net_one_layer_with_hardcoded_nonlinearity(nn.Module):
    def __init__(self, size_layer = 200):
        super().__init__()
        self.size_layer = size_layer
        self.fc1 = nn.Linear(1, self.size_layer)
        torch.nn.init.uniform_(self.fc1.bias,a=-1.0, b = 1.0)
        torch.nn.init.uniform_(self.fc1.weight,a=-1.0, b =2.0)

        self.fc2 = nn.Linear(self.size_layer, 1)
        torch.nn.init.normal_(self.fc2.weight,mean=2/self.size_layer, std = 0.01)
        torch.nn.init.normal_(self.fc2.bias,mean=0.0, std = 0.001)

        self.hardcoded_nonlinearity = nn.Linear(1, 1)
        torch.nn.init.normal_(self.hardcoded_nonlinearity.weight,mean=1.0, std = 0.000001)
        torch.nn.init.normal_(self.hardcoded_nonlinearity.bias,mean=-0.8, std = 0.000001)
        self.hardcoded_nonlinearity.weight.requires_grad = False
        self.hardcoded_nonlinearity.weight.requires_grad = False

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        y = y + F.relu(self.hardcoded_nonlinearity(x))
        return y

class Net_two_layers(nn.Module):
    def __init__(self, size_layer = 200):
        super().__init__()
        self.size_layer = size_layer
        self.fc1 = nn.Linear(1, self.size_layer)
        torch.nn.init.uniform_(self.fc1.bias,a=-1.0, b = 1.0)
        torch.nn.init.uniform_(self.fc1.weight,a=-1.0, b =2.0)

        self.fc2 = nn.Linear(self.size_layer, self.size_layer)
        torch.nn.init.normal_(self.fc2.weight,mean=2/self.size_layer, std = 0.01)
        torch.nn.init.normal_(self.fc2.bias,mean=0.0, std = 0.001)
        self.fc2.weight[0][0] = 1.0
        self.fc2.bias[0][0] = -0.8
        self.fc3 = nn.Linear(self.size_layer, 1)
        torch.nn.init.normal_(self.fc2.weight,mean=2/self.size_layer, std = 0.01)
        torch.nn.init.normal_(self.fc2.bias,mean=0.0, std = 0.001)


    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y

class Net_simple_neuron(nn.Module):
    def __init__(self, size_layer = 1,init_a = 1.0,bias_trainable=True):
        super().__init__()
        self.size_layer = size_layer
        self.fc1 = nn.Linear(1, self.size_layer)
        torch.nn.init.normal_(self.fc1.bias,mean= 0.0, std = 0.000001)
        self.fc1.bias.requires_grad=bias_trainable
        torch.nn.init.normal_(self.fc1.weight,mean=init_a, std =0.000001)

    def forward(self, x):
        y = self.fc1(x)
        return y

def optimal_strategy_with_positive_virtual_value(input,r):
        target = torch.nn.Parameter(r,requires_grad=False)*(1.00000001 - torch.nn.Parameter(r,requires_grad=False))/(torch.tensor(1.00000001)-input)
        return target

class Net_thresholded(nn.Module):
    def __init__(self, size_layer = 100):
        super().__init__()
        self.r = torch.nn.Parameter(torch.Tensor([0.5]),requires_grad=True)
        self.size_layer = size_layer
        #self.fc1 = nn.Linear(1, self.size_layer)
        #torch.nn.init.uniform_(self.fc1.bias,a=-1.0, b = 1.0)
        #torch.nn.init.uniform_(self.fc1.weight,a=-1.0, b =2.0)

        #self.fc2 = nn.Linear(self.size_layer, 1)
        #torch.nn.init.normal_(self.fc2.weight,mean=2/self.size_layer, std = 0.01)
        #torch.nn.init.normal_(self.fc2.bias,mean=0.0, std = 0.001)

        self.size_layer = size_layer
        self.fc1 = nn.Linear(1, self.size_layer)
        torch.nn.init.normal_(self.fc1.bias,mean= 0.0, std = 0.000001)
        self.fc1.bias.requires_grad=False
        torch.nn.init.normal_(self.fc1.weight,mean=1.0, std =0.000001)

        #self.r = nn.parameters()

    def forward(self, x):
        #y = self.fc2(F.relu(self.fc1(x)))*torch.sigmoid(100*(x-self.r))\
        #    +optimal_strategy_with_positive_virtual_value(x,self.r)*torch.sigmoid(100*(self.r-x))
        #y = torch.sigmoid(100*(x-self.r))*self.fc1(x)
        y = self.fc1(x)*torch.sigmoid(100000*(x-self.r))\
            +optimal_strategy_with_positive_virtual_value(x,self.r)*torch.sigmoid(100000*(self.r-x))
        return y


def main(net,loss_function,nb_steps=500,lr=0.0001,size_batch=2500):
    loss_list = []
    optimizer = optim.SGD(net.parameters(), lr=lr)
    print(net.parameters)

    for i in range(nb_steps):
        if i == int(nb_steps/2):
            lr/=10
            size_batch*=2
        input = torch.zeros((size_batch,1),requires_grad=True)
        samples = loss_function.distrib.sample((size_batch,1))
        input.data = samples.clone()
        loss = loss_function.eval(net,input,size_batch)
        loss_eval = -loss.detach().numpy()
        loss_list.append(loss_eval)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()    # Does the update
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f"loss after {i} iterations: {loss_eval}")
            if loss_function.name == "BoostedSecondPriceAffineFit":
                utils.compute_affine_regression(input,net,loss_function.distrib)
            if loss_function.name == "BoostedSecondPriceLinearFit":
                utils.compute_linear_regression(input,net,loss_function.distrib)
        #if i % 10:
            #utils.plot_virtual_value(net,loss_eval)


    utils.plot_loss(nb_steps,loss_list)
    utils.plot_strategy(net,loss_function.distrib)
    utils.plot_virtual_value(net,loss_function.distrib)
    return net

if __name__ == '__main__':
    print("You need to define a net and a loss")
