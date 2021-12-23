import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import time


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


def train(device, train_loader, netG, netD, num_epochs, batch_size, noise_size, num_classes):
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(num_classes)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        print(train_loader)
        for i, (images, labels) in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_images = images.to(device)
            labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_images, labels_one_hot).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise, labels_one_hot)
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            # Classify all fake batch with D
            output = netD(fake.detach(), labels_one_hot.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_labels)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            netD.optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, labels_one_hot).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, real_labels)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            netG.optimizer.step()

            # Output training stats and save model
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                path = 'models/gan_checkpoint'
                torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                }, path)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

    # Save model
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = './models/gan_' + time_stamp
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, path)
