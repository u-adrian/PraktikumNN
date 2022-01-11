from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xaviernormal(m.weight.data, gain=1.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.xaviernormal(m.weight.data, gain=1.0)
        nn.init.constant_(m.bias.data, 0)
