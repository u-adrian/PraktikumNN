import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict


class Conv2dAuto(nn.Conv2d):
    """
    Dynamically adds padding to the convolution based on the kernel size
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ResidualBlock(nn.Module):
    """
    Basic residual structure. If dimensions change -> residual needs to be rescaled via shortcut
    """
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.activate = activation

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    """
    Implementation of the shortcut for the residual
    """
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1,
                 conv=partial(Conv2dAuto, kernel_size=3, bias=False), *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Implementation of the actions inside a block
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation,
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetLayer(nn.Module):
    """
    Defines a stack of residual blocks
    The first Block handles downsampling by changing the stride of the convolution
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    Defines a composition of a gate and a stack of layers with increasing size in channels
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2, 2, 2, 2],
                 activation=nn.ReLU(), block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.gate_size = int(self.blocks_sizes[0] / 2)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.gate_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.gate_size),
            activation,
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(self.gate_size, blocks_sizes[0], n=depths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    Implementation of the decoding of features into one output value
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = self.activation(x)
        return x


class ResNet(nn.Module):
    """
    Defines a ResNet composed of encoder and decoder
    Also handles one hot label reshaping
    """
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, input_images, labels):
        labels = labels[:, :, None, None]  # shape: [batch_size, num_classes, 1, 1]
        label_maps = labels.expand(-1, -1, 32, 32).float()  # shape: [batch_size, num_classes, 32, 32]
        x = torch.cat((input_images, label_maps), 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnetDiscriminatorDepth1(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, blocks_sizes=[128, 256, 512], depths=[1, 1, 1])


def resnetDiscriminatorDepth2(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, blocks_sizes=[128, 256, 512], depths=[2, 2, 2])


def resnetDiscriminatorDepth1Leaky(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, blocks_sizes=[128, 256, 512], depths=[1, 1, 1],
                  activation=nn.LeakyReLU(0.2))
