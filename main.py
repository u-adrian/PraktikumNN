import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.optim as optim

import ResNetGanTraining
import ResNetGenerator
import ResNetDiscriminator

import GAN_v1_adrian
import random


def load_cifar10(batch_size):
    # fix download error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_v1_adrian():
    # Set hyper parameter
    noise_size = 100
    num_classes = 10
    image_shape = (3, 32, 32)
    n_image_channels = image_shape[0]
    batch_size = 100
    num_epochs = 5
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get nets
    netG = GAN_v1_adrian.GeneratorNet(noise_size, num_classes, n_image_channels, learning_rate, betas).to(device)
    netG.apply(GAN_v1_adrian.weights_init)
    netD = GAN_v1_adrian.DiscriminatorNet(n_image_channels, learning_rate, betas).to(device)
    netD.apply(GAN_v1_adrian.weights_init)

    # get data
    train_loader, test_loader = load_cifar10(batch_size)

    # start training
    GAN_v1_adrian.train(device, train_loader, netG, netD, num_epochs, batch_size, noise_size, num_classes)


def load_gan(path, netG, netD):
    checkpoint = torch.load(path)

    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])

    return netG, netD


def test_v1_adrian():
    # Set hyper parameter
    noise_size = 100
    num_classes = 10
    image_shape = (3, 32, 32)
    n_image_channels = image_shape[0]
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get nets
    netG = GAN_v1_adrian.GeneratorNet(noise_size, num_classes, n_image_channels, learning_rate, betas).to(device)
    netD = GAN_v1_adrian.DiscriminatorNet(n_image_channels, learning_rate, betas).to(device)

    # get weights
    path = './models/gan_2021-12-08_14-27-44'
    netG, netD = load_gan(path, netG, netD)
    netG.to(device)
    netD.to(device)

    # initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(num_classes)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    # create label as one hot
    labels = torch.tensor([range(10)])
    labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)

    # generate noise and stack 10 copies of that noise
    noise = torch.randn(1, noise_size, 1, 1, device=device)
    noise = noise.repeat(10, 1, 1, 1)

    # give to generator and denormalize output
    fake = netG(noise, labels_one_hot)
    # as info:
    # normalize = T.Normalize(mean.tolist(), std.tolist())
    # denormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    transform = transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))
    fake = transform(fake)

    for i, f in enumerate(fake):
        # show output
        output = f.cpu().detach().numpy()
        image = np.transpose(output, (1, 2, 0))
        plt.imshow(image)
        plt.title(f'Class: {classes[i]}')
        plt.show()


def test_resnet_discriminator():
    model = ResNetDiscriminator.resnetDiscriminator(3, 1)
    model.cuda()
    summary(model, (3, 32, 32))
    # print(model)

    image = torch.randn(100, 3, 32, 32).cuda()

    output = model(image)

    print(output.shape)


def test_discriminator(model_path):
    # Set hyper parameter
    noise_size = 100
    num_classes = 10
    image_channels = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 10

    # Get models
    model_g = ResNetGenerator.resnet18T(noise_size + num_classes, image_channels).to(device)
    model_d = ResNetDiscriminator.resnet18(image_channels + num_classes, 1).to(device)

    # get weights
    path = model_path
    model_g, model_d = load_gan(path, model_g, model_d)
    model_g.to(device)
    model_d.to(device)

    # Get data
    train_loader, test_loader = load_cifar10(batch_size)

    for i, (images, labels) in enumerate(test_loader):
        label_maps = torch.zeros(batch_size, num_classes, 32, 32)
        for j in range(batch_size):  # TODO: solve without loop
            label_maps[j, labels[j]] = torch.ones(32, 32)
        real_batch = torch.cat((images, label_maps), 1).to(device)
        # Forward pass through discriminator
        output_real = model_d(real_batch).view(-1)
        for j in range(batch_size):
            print('hoped for  1 but got: ', output_real[j])

        wrong_label_maps = torch.zeros(batch_size, num_classes, 32, 32)
        for j in range(batch_size):  # TODO: solve without loop
            random_wrong_label = random.choice([x for x in range(num_classes) if x != labels[j]])
            wrong_label_maps[j, random_wrong_label] = torch.ones(32, 32)
        real_wrong_batch = torch.cat((images, wrong_label_maps), 1).to(device)
        # Forward pass through discriminator
        output_real_wrong = model_d(real_wrong_batch).view(-1)
        for j in range(batch_size):
            print('hoped for  0 but got: ', output_real_wrong[j])

        break


def test_resnet_generator():
    model = ResNetGenerator.resnetGenerator(100+10, 3)
    model.cuda()
    summary(model, (100+10, 1, 1))
    # print(model)

    image = torch.randn(100, 100+10, 1, 1).cuda()

    output = model(image)

    print(f"Input: {image.shape}")
    print(f"Output: {output.shape}")


def train_resnet_gan():
    # Set hyper parameter
    noise_size = 100
    num_classes = 10
    image_channels = 3
    batch_size = 100
    num_epochs = 100
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data
    train_loader, _ = load_cifar10(batch_size)

    # Get models
    model_g = ResNetGenerator.resnet18T(noise_size + num_classes, image_channels).to(device)
    model_g.optimizer = optim.Adam(model_g.parameters(), lr=learning_rate, betas=betas)
    model_g.apply(ResNetGanTraining.weights_init)

    model_d = ResNetDiscriminator.resnet18(image_channels + num_classes, 1).to(device)
    model_d.optimizer = optim.Adam(model_d.parameters(), lr=learning_rate, betas=betas)
    model_d.apply(ResNetGanTraining.weights_init)

    # Start training
    ResNetGanTraining.train(device, train_loader, model_g, model_d, batch_size, num_epochs, noise_size, num_classes)


def test_resnet_gan(model_path):
    # Set hyper parameter
    noise_size = 100
    num_classes = 10
    image_channels = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get models
    model_g = ResNetGenerator.resnet18T(noise_size + num_classes, image_channels).to(device)
    model_d = ResNetDiscriminator.resnet18(image_channels + num_classes, 1).to(device)

    # get weights
    path = model_path
    model_g, model_d = load_gan(path, model_g, model_d)
    model_g.to(device)
    model_d.to(device)

    # initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(num_classes)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    # generate noise and stack 10 copies of that noise
    noise = torch.randn(1, noise_size, 1, 1, device=device)
    noise = noise.repeat(10, 1, 1, 1)

    # create label as one hot
    labels = torch.tensor([range(10)])
    labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)
    labels_one_hot = labels_one_hot[:, :, None, None]
    latent_vectors = torch.cat((noise, labels_one_hot), 1).to(device)

    # Generate batch (fake images + desired classes)
    fake_images = model_g(latent_vectors.float())

    # as info:
    # normalize = T.Normalize(mean.tolist(), std.tolist())
    # denormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    transform = transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))
    fake = transform(fake_images)

    for i, f in enumerate(fake):
        # show output
        output = f.cpu().detach().numpy()
        image = np.transpose(output, (1, 2, 0))
        plt.imshow(image)
        plt.title(f'Class: {classes[i]}')
        plt.show()


def main():
    # train_v1_adrian()
    # test_v1_adrian()
    # test_resnet_discriminator()
    test_discriminator('models/gan_2021-12-15_16-40-00')
    # test_resnet_generator()
    # train_resnet_gan()
    # test_resnet_gan('models/gan_2021-12-15_16-40-00')


if __name__ == "__main__":
    main()
