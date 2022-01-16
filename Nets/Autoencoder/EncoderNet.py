import torch
import torch.nn as nn

class EncoderNet(nn.Module):
    def __init__(self, n_image_channels, noise_size):
        super(EncoderNet, self).__init__()

        n_feature_maps = 64

        self.main = nn.Sequential(
            # input is (image_channels) x 32 x 32
            nn.Conv2d(n_image_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            # state size. (n_feature_maps) x 16 x 16
            nn.Conv2d(n_feature_maps, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.ReLU(inplace=True),
            # state size. (n_feature_maps*2) x 8 x 8
            nn.Conv2d(n_feature_maps * 2, noise_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(noise_size),
            nn.ReLU(inplace=True),
            # state size. noise_size x 4 x 4
            nn.MaxPool2d(4, 1, 0)
            # state size noise_size x 1 x 1
            #nn.Flatten()
            # state size 20
        )

    def forward(self, input_image: torch.Tensor,) -> torch.Tensor:
        return self.main(input_image)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
