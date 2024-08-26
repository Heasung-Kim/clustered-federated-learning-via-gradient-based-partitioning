import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

class LinearRegressor(nn.Module):
    # 1D Linear Regressor

    def __init__(self):

        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(1,1)
        #self.linear.weight.data = torch.from_numpy(np.tan(np.random.uniform(low=0, high=np.pi/2,size=(1,1))).astype(np.float32))
        #self.linear.weight.data = torch.from_numpy((np.random.uniform(low=-1.6, high=1.6, size=(1,1))).astype(np.float32))
        self.linear.weight.data = torch.from_numpy((np.random.uniform(low=-0.8, high=0.8, size=(1,1))).astype(np.float32))
        self.linear.bias.data.uniform_(-0.0 , 0.0 )
        print("linear:", self.linear.weight.data, self.linear.bias.data)
    def forward(self, x):
        # x = self.weight * x + self.bias
        x = self.linear(x)
        return x
