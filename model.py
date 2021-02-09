import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.activation1 = nn.LeakyReLU(0.2)  # maybe leaky relu ?
        self.activation2 = nn.Sigmoid()

    def forward(self, x):

        x = self.activation1(self.conv1(x))
        x = self.activation1(self.conv2_bn(self.conv2(x)))
        x = self.activation1(self.conv3_bn(self.conv3(x)))
        x = self.activation1(self.conv4_bn(self.conv4(x)))
        x = self.activation2(self.conv5(x))
        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.trconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.trconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.trconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.trconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.trconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1,  bias=False)
        self.activation1 = nn.ReLU()  # maybe leaky relu ?
        self.activation2 = nn.Tanh()  # why do they all use this

    def forward(self, x):
        # Convolutional layers
        x = self.activation1(self.conv1_bn(self.trconv1(x)))
        x = self.activation1(self.conv2_bn(self.trconv2(x)))
        x = self.activation1(self.conv3_bn(self.trconv3(x)))
        x = self.activation1(self.conv4_bn(self.trconv4(x)))
        x = self.activation2(self.trconv5(x))
        return x

