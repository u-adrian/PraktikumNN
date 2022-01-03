import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualTBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, contraction=1, upsampling=1, kernel_size=3,
                 conv=partial(nn.ConvTranspose2d, kernel_size=3, stride=1, padding=1, bias=False), *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.contraction, self.upsampling, self.conv, self.kernel_size = contraction, upsampling, conv, kernel_size

        # Resize residual to fit new size: half channels, double size
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.ConvTranspose2d(self.in_channels, self.contracted_channels, kernel_size=2,
                                           stride=self.upsampling, padding=0, bias=False),
                'bn': nn.BatchNorm2d(self.contracted_channels)

            })) if self.should_apply_shortcut else None

    @property
    def contracted_channels(self):
        return int(self.out_channels / self.contraction)

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.contracted_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


class ResNetBasicTBlock(ResNetResidualTBlock):
    contraction = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False,
                    stride=self.upsampling, kernel_size=self.kernel_size),
            activation(),
            conv_bn(self.out_channels, self.contracted_channels, conv=self.conv, bias=False),
        )


class ResNetTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicTBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform upsampling directly by transposed convolutional layers that have a stride of 2 and kernel size 4.'
        upsampling = 2 if in_channels != out_channels else 1
        kernel_size = 4 if in_channels != out_channels else 3

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, upsampling=upsampling, kernel_size=kernel_size),
            *[block(out_channels * block.contraction,
                    out_channels, upsampling=1, kernel_size=3, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetTEncoder(nn.Module):
    """
    ResNet Transposed encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels=100+10, blocks_sizes=[512, 256, 128, 64], depths=[2, 2, 2, 2],
                 activation=nn.ReLU, block=ResNetBasicTBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.ConvTranspose2d(in_channels, self.blocks_sizes[0], kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetTLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                         block=block, *args, **kwargs),
            *[ResNetTLayer(in_channels * block.contraction,
                           out_channels, n=n, activation=activation,
                           block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNetTDecoder(nn.Module):
    """
    fit to image size
    """

    def __init__(self, in_channels, image_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, image_channels, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x


class ResNetTransposed(nn.Module):

    def __init__(self, in_channels, image_channels, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetTEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetTDecoder(self.encoder.blocks[-1].blocks[-1].contracted_channels, image_channels)

    def forward(self, noise, labels_one_hot):
        labels_one_hot = labels_one_hot[:, :, None, None].float()
        x = torch.cat((noise, labels_one_hot), 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnetGenerator(in_channels, n_classes):
    return ResNetTransposed(in_channels, n_classes, block=ResNetBasicTBlock, depths=[1, 1, 1, 1])


def resnet18T(in_channels, n_classes):
    return ResNetTransposed(in_channels, n_classes, block=ResNetBasicTBlock, depths=[2, 2, 2, 2])