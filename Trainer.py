import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import time

from torch import optim

import CustumExceptions
import Data_Loader
import ResNetGanTraining
from Nets import Utils
from Nets.ResNet import ResNetGenerator, ResNetDiscriminator
from Nets.SmallGan import Small_GAN


###training method###
def train(**kwargs):
    # handle all keyword argument tuples:

    ####CONSTANTS####
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3

    #### variables ####
    #TODO: DEFAULT WERTE
    device = None
    generator = None
    discriminator = None
    num_epochs = 5
    batch_size = 100
    criterion = None
    real_img_fake_label = False
    noise_size = 100
    learning_rate = 0.0002
    betas = (0.5, 0.999) #TODO

    # Handle Device
    if 'device' in kwargs:
        if kwargs['device'] == "GPU":
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise CustumExceptions.GpuNotFoundError("Cannot find a CUDA device")
        else:
            device = torch.device('cpu')

    # Handle learning rate
    if 'learning_rate' in kwargs:
        try:
            learning_rate = float(kwargs['learning_rate'])
        except ValueError:
            raise CustumExceptions.LearningRateError("The learning rate must be float")
    else:
        print(f'Learning rate is not defined. Will use the default value "{learning_rate}" instead')

    # Handle noise size
    if 'noise_size' in kwargs:
        try:
            noise_size = int(kwargs['noise_size'])
            if batch_size <= 0:
                raise CustumExceptions.InvalidNoiseSizeError("noise_size must be greater than 0.")
        except ValueError:
            raise CustumExceptions.InvalidNoiseSizeError("noise_size must be a positive integer")
    else:
        print(f'noise_size is not defined. Will use the default value "{noise_size}" instead')

    # Handle Generator Net
    if 'generator' in kwargs:
        if kwargs['generator'] == "small_gan":
            generator = Small_GAN.GeneratorNet(noise_size=noise_size, num_classes=NUM_CLASSES, n_image_channels=N_IMAGE_CHANNELS,
                                               learning_rate=learning_rate, betas=betas).to(device)
            generator.apply(Utils.weights_init)
        elif kwargs['generator'] == "resNet":
            generator = ResNetGenerator.resnet18T(noise_size + NUM_CLASSES, N_IMAGE_CHANNELS).to(device)
            generator.optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
            generator.apply(ResNetGanTraining.weights_init)
        else:
            raise CustumExceptions.NoGeneratorError(f'The given generator net "{kwargs["discriminator"]}" cannot be found')
    else:
        raise CustumExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")

    # Handle Discriminator Net
    if 'discriminator' in kwargs:
        if kwargs['discriminator'] == "small_gan":
            discriminator = Small_GAN.DiscriminatorNet(n_image_channels=N_IMAGE_CHANNELS, learning_rate=learning_rate,
                                                       betas=betas).to(device)
            discriminator.apply(Utils.weights_init)
        elif kwargs['discriminator'] == "resNet":
            discriminator = ResNetDiscriminator.resnet18(N_IMAGE_CHANNELS + NUM_CLASSES, 1).to(device)
            discriminator.optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
            discriminator.apply(ResNetGanTraining.weights_init)
            #raise NotImplementedError
        else:
            raise CustumExceptions.NoDiscriminatorError(f'The given discriminator net "{kwargs["discriminator"]}" cannot be found')
    else:
        raise CustumExceptions.NoDiscriminatorError(
            "You need to define the discriminator net. keyword: 'discriminator'")

    # Handle num_epochs
    if 'num_epochs' in kwargs:
        try:
            num_epochs = int(kwargs['num_epochs'])
            if num_epochs <= 0:
                raise CustumExceptions.NumEpochsError("The Number of epochs must be greater than 0")
        except ValueError:
            raise CustumExceptions.NumEpochsError("The Number of epochs must be a positive integer")
    else:
        print(f'The number of epochs is not defined. Will use the default value "{num_epochs}" instead')

    # Handle batch size
    if 'batch_size' in kwargs:
        try:
            batch_size = int(kwargs['batch_size'])
            if batch_size <= 0:
                raise CustumExceptions.BatchSizeError("The batch size must be greater than 0!")
        except ValueError:
            raise CustumExceptions.BatchSizeError("The batch size must be a positive integer")
    else:
        print(f'The batch size is not defined. Will use the default value "{batch_size}" instead')

    # Handle criterion
    if 'criterion' in kwargs:
        if kwargs['criterion'] == 'BCELoss':
            criterion = nn.BCELoss()
        elif kwargs['criterion'] == 'Wasserstein':
            criterion = 'Wasserstein' #TODO
            raise NotImplementedError
        elif kwargs['criterion'] == 'MiniMax':
            criterion = 'MiniMax' #TODO
            raise NotImplementedError
        else:
            raise CustumExceptions.InvalidLossError()
    else:
        raise CustumExceptions.InvalidLossError()

    if 'real_img_fake_label' in kwargs:
        if kwargs['real_img_fake_label'].lower() in ['true','t','yes','y','1']:
            real_img_fake_label = True
        elif kwargs['real_img_fake_label'].lower() in ['false','f','no','n','0']:
            real_img_fake_label = False
        else:
            raise CustumExceptions.InvalidArgumentError(f'Invalid argument for "real_img_fake_label": "{kwargs["real_img_fake_label"]}"')
    else:
        print(f'No argument for "real_img_fake_label" found. Will use {real_img_fake_label}')


    #load dataset
    train_loader, _ = Data_Loader.load_cifar10(batch_size)

    # start training process
    __train(device=device, train_loader=train_loader, netG=generator, netD=discriminator, num_epochs=num_epochs,
            batch_size=batch_size, noise_size=noise_size, real_image_fake_label=real_img_fake_label)


def __train(device, train_loader, netG, netD, num_epochs, batch_size, noise_size, real_image_fake_label=False, num_classes=10):
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
            # (1) Update Discriminator network
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            real_images = images.to(device)
            labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images, labels_one_hot.detach()).view(-1)
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
            fake = netG(noise, labels_one_hot)
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            output = netD(fake.detach(), labels_one_hot.detach()).view(-1)
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            # Train with real images and fake labels (rifl)
            if real_image_fake_label:
                deviation_labels = labels + torch.randint(low=1, high=10, size=labels.shape)
                deviation_labels = torch.remainder(deviation_labels, 10)
                deviation_one_hot = torch.tensor(one_hot_enc.transform(deviation_labels.reshape(-1, 1)).toarray(),
                                                 device=device)

                output = netD(real_images, deviation_one_hot).view(-1)
                errD_fake_labels = criterion(output, fake_labels)
                errD_fake_labels.backward()

            # update the discriminator net
            netD.optimizer.step()

            ############################
            # (2) Update Generator network
            ###########################
            netG.zero_grad()
            output = netD(fake, labels_one_hot).view(-1)
            errG = criterion(output, real_labels)
            errG.backward()
            D_G_z2 = output.mean().item()
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
