'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, no_bn=False):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, bias=True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.no_bn = no_bn

    def forward(self, x):
        x = self.conv(x)
        if not self.no_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

class AConvNet(nn.Module):
    def __init__(self, in_channel, num_class=10, dropout_rate=0.1):
        super(AConvNet, self).__init__()

        self.num_class = num_class
        self.in_channel = in_channel

        self.layer1 = ConvBNReLU(in_channel, 16, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = ConvBNReLU(16, 32, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = ConvBNReLU(32, 64, 6)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = ConvBNReLU(64, 128, 5)
        self.layer5 = ConvBNReLU(128, num_class, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.reshape(-1, self.num_class)
        return x
