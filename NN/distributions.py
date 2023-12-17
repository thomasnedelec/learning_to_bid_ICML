import torch
from torch.distributions import Exponential

class UniformDistrib():
    def __init__(self):
        self.name="uniform"
        self.cdf=lambda x : x
        self.pdf=lambda x:1
        self.inverse_virtual_value = lambda x:(x+1)/2
        self.virtual_value = lambda x:2*x-1
        self.boost = 2.0
        self.optimal_reserve_price = 0.5
    def sample(self,size):
        return torch.rand(size)
class ExponentialDistrib():
    def __init__(self,lambdap=1.0):
        self.name="exponential"
        self.cdf = lambda x : 1 - torch.exp(-lambdap*x)
        self.pdf = lambda x : lambdap*torch.exp(-lambdap*x)
        self.inverse_virtual_value = lambda x : x+lambdap
        self.virtual_value = lambda x : x-lambdap
        self.boost = 1
        self.optimal_reserve_price = lambdap
    def sample(self,size):
        m = Exponential(torch.tensor([1.0]))
        return m.sample(size)
