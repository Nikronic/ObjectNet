import os
import sys
import torch
import torch.nn as nn
import math
from urllib.request import urlretrieve  # if does not work, use this: from urllib import urlretrieve

# url to pretrained ResNet-101 Model
model_url = 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'


def resnet101(pretrained=False, **kwargs):
    """
    A ResNet-101 model

    :param pretrained: Use pretrained model on places(standard) or not
    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_url), strict=False)
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannel, channel, stride=1, downsample=None):
        """

        :param inchannel: input channel of block
        :param channel: channel size in mid. 4x at output
        :param stride: stride
        :param downsample: use downsampling
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, channel, kernel_size=1, bias=False, stride=1)
        self.bn1 = None
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = None
        self.conv3 = nn.Conv2d(channel, channel * 4, kernel_size=1, bias=False)
        self.bn3 = None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
