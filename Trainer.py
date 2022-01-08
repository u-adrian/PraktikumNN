from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import time

from torch import optim

import ArgHandler
import CustomExceptions
import Data_Loader
from Nets import Utils
from Nets.ResNet import ResNetGenerator, ResNetDiscriminator
from Nets.SmallGan import Small_GAN


def train(**kwargs):
    t = Trainer(**kwargs)
    t.train()
    return t.unique_name


class Trainer:
    # CONSTANT
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3

    # VARIABLES
    name = None
    unique_name = None

    do_snapshots = False
    snapshot_interval = -1

    device = None
    generator = None
    discriminator = None
    num_epochs = 5
    batch_size = 100
    criterion = None
    real_img_fake_label = False
    noise_size = 100
    learning_rate = 0.0002
    betas = (0.5, 0.999)  # TODO

    def __init__(self, **kwargs):
        self.__parse_args(**kwargs)
        self.__create_folder_structure()

    def train(self):
        train_loader, _ = Data_Loader.load_cifar10(self.batch_size)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # initialize One Hot encoder
        one_hot_enc = OneHotEncoder()
        all_classes = torch.tensor(range(self.NUM_CLASSES)).reshape(-1, 1)
        one_hot_enc.fit(all_classes)

        # Lists to keep track of progress
        G_losses = []
        D_losses = []

        # Training Loop
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            print(train_loader)
            for i, (images, labels) in enumerate(train_loader, 0):
                ############################
                # (1) Update Discriminator network
                ###########################
                # Train with all-real batch
                self.discriminator.zero_grad()
                real_images = images.to(self.device)
                labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=self.device)
                real_labels = torch.full((self.batch_size,), real_label, dtype=torch.float, device=self.device)
                output = self.discriminator(real_images, labels_one_hot.detach()).view(-1)
                errD_real = self.criterion(output, real_labels)
                errD_real.backward(retain_graph=True)
                D_x = output.mean().item()

                # Train with all-fake batch
                noise = torch.randn(self.batch_size, self.noise_size, 1, 1, device=self.device)
                fake = self.generator(noise, labels_one_hot)
                fake_labels = torch.full((self.batch_size,), fake_label, dtype=torch.float, device=self.device)
                output = self.discriminator(fake.detach(), labels_one_hot.detach()).view(-1)
                errD_fake = self.criterion(output, fake_labels)
                errD_fake.backward(retain_graph=True)
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake

                # Train with real images and fake labels (rifl)
                if self.real_img_fake_label:
                    deviation_labels = labels + torch.randint(low=1, high=10, size=labels.shape)
                    deviation_labels = torch.remainder(deviation_labels, 10)
                    deviation_one_hot = torch.tensor(one_hot_enc.transform(deviation_labels.reshape(-1, 1)).toarray(),
                                                     device=self.device)

                    output = self.discriminator(real_images, deviation_one_hot).view(-1)
                    errD_fake_labels = self.criterion(output, fake_labels)
                    errD_fake_labels.backward(retain_graph=True)

                # update the discriminator net
                self.discriminator.optimizer.step()

                ############################
                # (2) Update Generator network
                ###########################
                self.generator.zero_grad()
                output = self.discriminator(fake, labels_one_hot).view(-1)
                errG = self.criterion(output, real_labels)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.generator.optimizer.step()

                # Output training stats and save model
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.num_epochs, i, len(train_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            if self.do_snapshots and self.snapshot_interval > 0 and epoch % self.snapshot_interval == 0:
                path = f'models/{self.unique_name}/snapshots/gan_after_epoch_{epoch}'
                torch.save({
                    'netG_state_dict': self.generator.state_dict(),
                    'netD_state_dict': self.discriminator.state_dict(),
                }, path)

        # Save model
        path = f'./models/{self.unique_name}/gan_latest'
        torch.save({
            'netG_state_dict': self.generator.state_dict(),
            'netD_state_dict': self.discriminator.state_dict(),
        }, path)

    def __create_folder_structure(self):
        Path(f'./models/{self.unique_name}/snapshots/').mkdir(parents=True, exist_ok=True)

    def __parse_args(self, **kwargs):
        # Handle Arguments
        self.device = ArgHandler.handle_device(**kwargs)

        self.learning_rate = ArgHandler.handle_learning_rate(**kwargs)

        self.noise_size = ArgHandler.handle_noise_size(**kwargs)

        self.generator = ArgHandler.handle_generator(self.NUM_CLASSES, self.N_IMAGE_CHANNELS, **kwargs)
        self.generator.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                              betas=self.betas)
        self.generator.apply(Utils.weights_init)

        self.discriminator = ArgHandler.handle_discriminator(self.NUM_CLASSES, self.N_IMAGE_CHANNELS, **kwargs)
        self.discriminator.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                                  betas=self.betas)
        self.discriminator.apply(Utils.weights_init)

        self.num_epochs = ArgHandler.handle_num_epochs(**kwargs)

        self.batch_size = ArgHandler.handle_batch_size(**kwargs)

        self.criterion = ArgHandler.handle_criterion(**kwargs)

        self.real_img_fake_label = ArgHandler.handle_real_img_fake_label(**kwargs)

        self.snapshot_interval, self.do_snapshots = ArgHandler.handle_snapshot_settings(**kwargs)

        self.name, self.unique_name = ArgHandler.handle_name(**kwargs)
