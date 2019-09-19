'''Encoder and Decoder both ResNet based.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Repository:
[1] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
[2] https://github.com/kuangliu/pytorch-cifar
'''

__author__ = 'JGB_Robosapien'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicDeBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicDeBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride,
                                          padding=1, output_padding =stride-1, bias=False)
        self.debn1 = nn.BatchNorm2d(planes)
        self.deconv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1,
                                          padding=1, bias=False)
        self.debn2 = nn.BatchNorm2d(planes)

        self.deshortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.deshortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,
                                   output_padding =stride-1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.debn1(self.deconv1(x)))
        out = self.debn2(self.deconv2(out))
        out += self.deshortcut(x)
        out = F.relu(out)
        return out


class ResEmbedNet(nn.Module):
    def __init__(self, block, num_blocks, inDim = [512,512] ,embedSize = 512):
        super(ResNet, self).__init__()

        if block  == BasicBlock:
            deblock = BasicDeBlock
        else:
            sys.stderr.write("only `BasicBlock` is implemented")

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 512
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 512
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 256
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 128
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 64 8x reduction
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.delayer4 = self._make_delayer(deblock, 512, num_blocks[3], stride=2)
        self.delayer3 = self._make_delayer(deblock, 256, num_blocks[2], stride=2)
        self.delayer2 = self._make_delayer(deblock, 128, num_blocks[1], stride=2)
        self.delayer1 = self._make_delayer(deblock, 64, num_blocks[0], stride=1)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.debn1 = nn.BatchNorm2d(3)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_delayer(self, deblock, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        delayers = []
        for stride in strides:
            delayers.append(deblock(self.in_planes, planes, stride))
            self.in_planes = planes * deblock.expansion
        return nn.Sequential(*delayers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        emb = self.avgpool(out)

        # emb1 = torch.tensor(emb).view(-1).cpu().numpy()

        out = self.delayer4(out)
        out = self.delayer3(out)
        out = self.delayer2(out)
        out = self.delayer1(out)

        out = torch.sigmoid(self.debn1(self.deconv1(out)))

        return out, emb
