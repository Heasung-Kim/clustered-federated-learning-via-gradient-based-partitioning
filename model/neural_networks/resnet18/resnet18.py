

import torch
import torch.nn as nn
from collections import OrderedDict

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        #self.channel_transformation = ConvBN(1,3,1)
        from torchvision.models import resnet18
        self.resnet18 = resnet18()
        #self.resnet18.fc.out_features = 100

        self.resnet18.fc = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        #x = self.channel_transformation(x)
        x = self.resnet18(x)
        #x = self.dense(x)
        return x


if __name__ == "__main__":
    config = {
        "img_shape": (3,32,32),
        "n_codeword_floats": 32
    }
    from torchvision import models
    #from torchsummary import summary

    crnet = Resnet18(config, device=None)
    print("hello")
    #summary(crnet, (1,24,24), device="cpu")
