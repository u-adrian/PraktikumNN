import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import time
import CustumExceptions

#### variables ####

###training method###
#TODO: add parameters: Loss function, use realImageFakeLabel (rIfL),

def train(**kwargs):
    # handle all keyword argument tuples:

    device = None
    generator = None
    discriminator = None
    num_epochs = None
    batch_size = None
    learning_rate = None
    criterion = None

    # Handle Devide
    if kwargs['device'] == "GPU":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise CustumExceptions.GpuNotFoundError("Cannot find a CUDA device")
    else:
        device = torch.device('cpu')

    # Handle Generator Net
    if kwargs['generator'] == "test":
        generator = "test"
        raise NotImplementedError
    elif kwargs['generator'] == "resNet":
        generator = "resNet"
        raise NotImplementedError
    else:
        raise CustumExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")

    # Handle Discriminator Net
    if kwargs['discriminator'] == "test":
        generator = "test"
        raise NotImplementedError
    elif kwargs['discriminator'] == "resNet":
        generator = "resNet"
        raise NotImplementedError
    else:
        raise CustumExceptions.NoDiscriminatorError("You need to define the discriminator net. keyword: 'discriminator'")

    # Handle num_epochs
    try:
        num_epochs = int(kwargs['num_epochs'])
        if num_epochs <= 0 or num_epochs > 20:
            raise CustumExceptions.NumEpochsError("The Number of epochs must be in the interval [1,20]!")
    except ValueError:
        raise CustumExceptions.NumEpochsError("The Number of epochs must be a positive integer")

    # Handle batch size
    try:
        batch_size = int(kwargs['batch_size'])
        if batch_size <= 0:
            raise CustumExceptions.BatchSizeError("The batch size must be greater than 0!")
    except ValueError:
        raise CustumExceptions.BatchSizeError("The batch size must be a positive integer")

    # Handle learning rate
    try:
        learning_rate = float(kwargs['learning_rate'])
    except ValueError:
        raise CustumExceptions.LearningRateError("The learning rate must be float")

    # Handle criterion
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




def _train(device, train_loader, netG, netD, num_epochs, batch_size, noise_size, num_classes=10):
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
            real_images = images.to(device)
            labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images, labels_one_hot).view(-1)
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
            netD.optimizer.step()

            # Train with real images and fake labels


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