import torch
import torch.nn as nn
import torch.optim as optim


class GeneratorNet(nn.Module):
    def __init__(self, noise_size, num_classes, n_image_channels, learning_rate, betas):
        super(GeneratorNet, self).__init__()
        # 10 classes
        # images 32x32 pixels ( = 1024 pixel)
        # noise_shape: 100x1x1
        input_length = noise_size + num_classes
        n_channels = 64
        # n_image_channels = image_shape[0]
        self.main = nn.Sequential(
            # input is noise and class as one-hot
            nn.ConvTranspose2d(input_length, n_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_channels * 8),
            nn.ReLU(True),
            # state size. (n_channels*8) x 4 x 4
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(True),
            # state size. (n_channels*4) x 8 x 8
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(True),
            # state size. (n_channels*2) x 16 x 16
            nn.ConvTranspose2d( n_channels * 2, n_image_channels, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_image_channels),
            nn.Tanh(),
            # state size. 3 x 32 x 32
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=betas)  # use adam optimizer

    def forward(self, noise: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = torch.unsqueeze(torch.unsqueeze(label, 2), 2)
        data = torch.cat((noise, label), 1).float()
        return self.main(data)


class DiscriminatorNet(nn.Module):
    def __init__(self, n_image_channels, learning_rate, betas):
        super(DiscriminatorNet, self).__init__()
        # 10 classes
        # images 32x32 pixels ( = 1024 pixel)
        # noise_shape: n x m

        # height = image_shape[1]
        # width = image_shape[2]
        # n_image_channels = image_shape[0]

        n_feature_maps = 64

        self.feature_extract = nn.Sequential(
            # input is (image_channels) x 32 x 32
            nn.Conv2d(n_image_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_feature_maps) x 16 x 16
            nn.Conv2d(n_feature_maps, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_feature_maps*2) x 8 x 8
            nn.Conv2d(n_feature_maps * 2, n_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_feature_maps*4) x 4 x 4
            nn.Conv2d(n_feature_maps * 4, 20, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=betas)  # use adam optimizer

    def forward(self, input_image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.feature_extract(input_image)
        data = torch.cat((features, labels), 1).float()
        return self.classifier(data)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
