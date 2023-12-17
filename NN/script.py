import bidder_strategy as learning
import importlib
import losses as losses
import utils
import torch
import matplotlib.pyplot as plt
importlib.reload(learning)
importlib.reload(losses)
importlib.reload(utils)
import numpy as np

net = learning.Net_one_layer()
loss_net = losses.lossBoostedSecondPrice(nb_opponents=1)
learning.main(net,loss_net,nb_steps=1000,size_batch=100000,lr = 0.0001)
