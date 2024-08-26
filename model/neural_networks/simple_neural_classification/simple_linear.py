"""

Reference : https://github.com/jichan3751/ifca/tree/main/mnist

"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleLinear(nn.Module):

    def __init__(self):
        h1 = 200
        super(SimpleLinear, self).__init__()
        self.fc1 = nn.Linear(28 * 28, h1)
        self.fc2 = nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    # def weight(self):
    #     return self.linear1.weight
