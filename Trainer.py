from pathlib import Path

import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import time

from torch import optim

import CustumExceptions
import Data_Loader
from Nets import Utils
from Nets.ResNet import ResNetGenerator, ResNetDiscriminator
from Nets.SmallGan import Small_GAN

def train(**kwargs):
    t = Trainer(**kwargs)
    t.train()

def train_and_get_uniquename(**kwargs):
    t = Trainer(**kwargs)
    t.train()
    return t.unique_name

class Trainer:
    ####CONSTANTS####
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3

    #### variables ####
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
        # Handle Device
        if 'device' in kwargs:
            if kwargs['device'] == "GPU":
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    raise CustumExceptions.GpuNotFoundError("Cannot find a CUDA device")
            else:
                self.device = torch.device('cpu')

        # Handle learning rate
        if 'learning_rate' in kwargs:
            try:
                self.learning_rate = float(kwargs['learning_rate'])
            except ValueError:
                raise CustumExceptions.LearningRateError("The learning rate must be float")
        else:
            print(f'Learning rate is not defined. Will use the default value "{self.learning_rate}" instead')

        # Handle noise size
        if 'noise_size' in kwargs:
            try:
                self.noise_size = int(kwargs['noise_size'])
                if self.noise_size <= 0:
                    raise CustumExceptions.InvalidNoiseSizeError("noise_size must be greater than 0.")
            except ValueError:
                raise CustumExceptions.InvalidNoiseSizeError("noise_size must be a positive integer")
        else:
            print(f'noise_size is not defined. Will use the default value "{self.noise_size}" instead')

        # Handle Generator Net
        if 'generator' in kwargs:
            if kwargs['generator'] == "small_gan":
                self.generator = Small_GAN.GeneratorNet(noise_size=self.noise_size, num_classes=self.NUM_CLASSES,
                                                        n_image_channels=self.N_IMAGE_CHANNELS).to(self.device)
                self.generator.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                                      betas=self.betas)
                self.generator.apply(Utils.weights_init)
            elif kwargs['generator'] == "res_net_depth1":
                self.generator = ResNetGenerator.resnetGeneratorDepth1(self.noise_size + self.NUM_CLASSES,
                                                                       self.N_IMAGE_CHANNELS).to(self.device)
                self.generator.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                                      betas=self.betas)
                self.generator.apply(Utils.weights_init)
            elif kwargs['generator'] == "res_net_depth2":
                self.generator = ResNetGenerator.resnetGeneratorDepth2(self.noise_size + self.NUM_CLASSES,
                                                                       self.N_IMAGE_CHANNELS).to(self.device)
                self.generator.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                                      betas=self.betas)
                self.generator.apply(Utils.weights_init)
            else:
                raise CustumExceptions.NoGeneratorError(
                    f'The given generator net "{kwargs["generator"]}" cannot be found')
        else:
            raise CustumExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")

        # Handle Discriminator Net
        if 'discriminator' in kwargs:
            if kwargs['discriminator'] == "small_gan":
                self.discriminator = Small_GAN.DiscriminatorNet(n_image_channels=self.N_IMAGE_CHANNELS,
                                                                num_classes=self.NUM_CLASSES).to(self.device)
                self.discriminator.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                                          betas=self.betas)
                self.discriminator.apply(Utils.weights_init)
            elif kwargs['discriminator'] == "res_net_depth1":
                self.discriminator = ResNetDiscriminator.resnetDiscriminatorDepth1(self.N_IMAGE_CHANNELS
                                                                                   + self.NUM_CLASSES, 1).to(self.device)
                self.discriminator.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                                          betas=self.betas)
                self.discriminator.apply(Utils.weights_init)
            elif kwargs['discriminator'] == "res_net_depth2":
                self.discriminator = ResNetDiscriminator.resnetDiscriminatorDepth2(self.N_IMAGE_CHANNELS
                                                                                   + self.NUM_CLASSES, 1).to(self.device)
                self.discriminator.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                                          betas=self.betas)
                self.discriminator.apply(Utils.weights_init)
            else:
                raise CustumExceptions.NoDiscriminatorError(
                    f'The given discriminator net "{kwargs["discriminator"]}" cannot be found')
        else:
            raise CustumExceptions.NoDiscriminatorError(
                "You need to define the discriminator net. keyword: 'discriminator'")

        # Handle num_epochs
        if 'num_epochs' in kwargs:
            try:
                self.num_epochs = int(kwargs['num_epochs'])
                if self.num_epochs <= 0:
                    raise CustumExceptions.NumEpochsError("The Number of epochs must be greater than 0")
            except ValueError:
                raise CustumExceptions.NumEpochsError("The Number of epochs must be a positive integer")
        else:
            print(f'The number of epochs is not defined. Will use the default value "{self.num_epochs}" instead')

        # Handle batch size
        if 'batch_size' in kwargs:
            try:
                self.batch_size = int(kwargs['batch_size'])
                if self.batch_size <= 0:
                    raise CustumExceptions.BatchSizeError("The batch size must be greater than 0!")
            except ValueError:
                raise CustumExceptions.BatchSizeError("The batch size must be a positive integer")
        else:
            print(f'The batch size is not defined. Will use the default value "{self.batch_size}" instead')

        # Handle criterion
        if 'criterion' in kwargs:
            if kwargs['criterion'] == 'BCELoss':
                self.criterion = nn.BCELoss()
            elif kwargs['criterion'] == 'Wasserstein':
                self.criterion = 'Wasserstein'  # TODO
                raise NotImplementedError
            elif kwargs['criterion'] == 'MiniMax':
                self.criterion = 'MiniMax'  # TODO
                raise NotImplementedError
            else:
                raise CustumExceptions.InvalidLossError()
        else:
            raise CustumExceptions.InvalidLossError()

        if 'real_img_fake_label' in kwargs:
            if kwargs['real_img_fake_label'].lower() in ['true', 't', 'yes', 'y', '1']:
                self.real_img_fake_label = True
            elif kwargs['real_img_fake_label'].lower() in ['false', 'f', 'no', 'n', '0']:
                self.real_img_fake_label = False
            else:
                raise CustumExceptions.InvalidArgumentError(
                    f'Invalid argument for "real_img_fake_label": "{kwargs["real_img_fake_label"]}"')
        else:
            print(f'No argument for "real_img_fake_label" found. Will use {self.real_img_fake_label}')

        # Handle snapshot settings
        if 'snapshot_interval' in kwargs:
            try:
                self.snapshot_interval = int(kwargs['snapshot_interval'])
                self.do_snapshots = True
                if self.snapshot_interval <= 0:
                    raise CustumExceptions.InvalidSnapshotInterval("snapshot_interval must be greater than 0.")
            except ValueError:
                raise CustumExceptions.InvalidNoiseSizeError("snapshot_interval must be a positive integer")
        else:
            self.do_snapshots = False
            print(f'snapshot_interval is not defined. Will not create nor store snapshots')

        # Handle name and unique name
        if 'name' in kwargs:
            self.name = kwargs['name']
            self.unique_name = f'{self.name}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
            if not self.name.isalnum():
                raise CustumExceptions.InvalidNameError("name must only contain alphanumeric characters")
        else:
            self.unique_name = time.strftime("%Y-%m-%d_%H-%M-%S")
            print("No name given. Will only use the time-stamp")
