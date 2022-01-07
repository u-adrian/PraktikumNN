from pathlib import Path

import torch
import torchvision.transforms as transforms

from sklearn.preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import CustomExceptions
from Nets.ResNet import ResNetGenerator
from Nets.SmallGan import Small_GAN

from Data_Loader import load_cifar10
from scores import inception_score
import os


def evaluate_model(**kwargs):
    #### CONSTANTS ####
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #### Variables ####
    generator = None
    device = None
    model_path = None
    output_path = None
    noise_size = None

    # Handle Device
    if 'device' in kwargs:
        if kwargs['device'] == "GPU":
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise CustomExceptions.GpuNotFoundError("Cannot find a CUDA device")
        else:
            device = torch.device('cpu')

    # Handle noise size
    if 'noise_size' in kwargs:
        try:
            noise_size = int(kwargs['noise_size'])
            if noise_size <= 0:
                raise CustomExceptions.InvalidNoiseSizeError("noise_size must be greater than 0.")
        except ValueError:
            raise CustomExceptions.InvalidNoiseSizeError("noise_size must be a positive integer")
    else:
        raise CustomExceptions.InvalidNoiseSizeError("You have to set the noise_size argument")

    # Handle Generator Net
    if 'generator' in kwargs:
        if kwargs['generator'] == "small_gan":
            generator = Small_GAN.GeneratorNet(noise_size=noise_size, num_classes=NUM_CLASSES,
                                               n_image_channels=N_IMAGE_CHANNELS).to(device)
        elif kwargs['generator'] == "res_net":
            generator = ResNetGenerator.resnetGenerator(noise_size + NUM_CLASSES, N_IMAGE_CHANNELS).to(device)
        else:
            raise CustomExceptions.NoGeneratorError(
                f'The given generator net "{kwargs["generator"]}" cannot be found')
    else:
        raise CustomExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")

    # Handle model_path
    if 'model_path' in kwargs:
        model_path = kwargs['model_path']
    else:
        raise NotImplementedError('Using a default model_path is not implemented. And will never be')

    # Handle model_path
    if 'output_path' in kwargs:
        output_path = kwargs['output_path']
    else:
        raise NotImplementedError('Using a default output_path is not implemented. And will never be')

    # Handle result_path
    if 'result_path' in kwargs:
        result_path = kwargs['result_path']
    else:
        raise NotImplementedError('Using a default result_path is not implemented. And will never be')

    # load generator
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['netG_state_dict'])

    # initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(NUM_CLASSES)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    batch_size=100

    _, test_loader = load_cifar10(batch_size)
    num_images = len(test_loader.dataset)
    fakes = torch.zeros([num_images, 3, 32, 32], dtype=torch.float32)
    for i, (images, labels) in enumerate(test_loader, 0):
        labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)
        noise = torch.randn(batch_size, noise_size, 1, 1, device=device)
        fake = generator(noise, labels_one_hot)
        batch_size_i = images.size()[0]
        fakes[i * batch_size:i * batch_size + batch_size_i] = fake
        print('Generating images: ', i * batch_size, '/', num_images)

    i_score = inception_score(fakes, device, 32)
    print('inception score: ', i_score)

    scores_file = f= open(result_path, "w+")
    scores_file.write("Inception_score: %d\r\n" % i_score)
    scores_file.close()








def create_images(**kwargs):
    #### CONSTANTS ####
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #### Variables ####
    generator = None
    device = None
    model_path = None
    output_path = None
    noise_size = None

    # Handle Device
    if 'device' in kwargs:
        if kwargs['device'] == "GPU":
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise CustomExceptions.GpuNotFoundError("Cannot find a CUDA device")
    else:
        device = torch.device('cpu')

    # Handle noise size
    if 'noise_size' in kwargs:
        try:
            noise_size = int(kwargs['noise_size'])
            if noise_size <= 0:
                raise CustomExceptions.InvalidNoiseSizeError("noise_size must be greater than 0.")
        except ValueError:
            raise CustomExceptions.InvalidNoiseSizeError("noise_size must be a positive integer")
    else:
        raise CustomExceptions.InvalidNoiseSizeError("You have to set the noise_size argument")

    # Handle Generator Net
    if 'generator' in kwargs:
        if kwargs['generator'] == "small_gan":
            generator = Small_GAN.GeneratorNet(noise_size=noise_size, num_classes=NUM_CLASSES,
                                               n_image_channels=N_IMAGE_CHANNELS).to(device)
        elif kwargs['generator'] == "res_net":
            generator = ResNetGenerator.resnetGenerator(noise_size + NUM_CLASSES, N_IMAGE_CHANNELS).to(device)
        else:
            raise CustomExceptions.NoGeneratorError(
                f'The given generator net "{kwargs["generator"]}" cannot be found')
    else:
        raise CustomExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")

    # Handle model_path
    if 'model_path' in kwargs:
        model_path = kwargs['model_path']
    else:
        raise NotImplementedError('Using a default model_path is not implemented. And will never be')

    # Handle model_path
    if 'output_path' in kwargs:
        output_path = kwargs['output_path']
    else:
        raise NotImplementedError('Using a default output_path is not implemented. And will never be')

    # load generator
    generator.load_state_dict(torch.load(model_path)['netG_state_dict'])

    # initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(NUM_CLASSES)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    # do it multiple times
    for j in range(10):

        # generate noise and stack 10 copies of that noise
        noise = torch.randn(1, noise_size, 1, 1, device=device)
        noise = noise.repeat(10, 1, 1, 1)

        # create label as one hot
        labels = torch.tensor([range(10)])
        labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)

        # Generate batch (fake images + desired classes)
        fake_images = generator(noise,labels_one_hot)
        print(fake_images.shape)



        # as info:
        # normalize = T.Normalize(mean.tolist(), std.tolist())
        # denormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        transform = transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5))
        fake = transform(fake_images)

        for i, f in enumerate(fake):
            # show output
            output = f.cpu().detach().numpy()
            image = np.transpose(output, (1, 2, 0))

            Path(output_path).mkdir(parents=True, exist_ok=True)
            im = Image.fromarray((image * 255).astype(np.uint8))
            im.save(output_path + f"/{classes[i]}_{j}.png")

            plt.imshow(image)
            plt.title(f'Class: {classes[i]}')
            # plt.show()

