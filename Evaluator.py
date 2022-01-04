import torch

import CustumExceptions
from Nets.ResNet import ResNetGenerator
from Nets.SmallGan import Small_GAN


def create_images(**kwargs):
    #### CONSTANTS ####
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3

    #### Variables ####
    generator = None
    device = None
    path = None
    noise_size = None

    # Handle Device
    if 'device' in kwargs:
        if kwargs['device'] == "GPU":
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise CustumExceptions.GpuNotFoundError("Cannot find a CUDA device")
        else:
            device = torch.device('cpu')

        # Handle noise size
        if 'noise_size' in kwargs:
            try:
                noise_size = int(kwargs['noise_size'])
                if noise_size <= 0:
                    raise CustumExceptions.InvalidNoiseSizeError("noise_size must be greater than 0.")
            except ValueError:
                raise CustumExceptions.InvalidNoiseSizeError("noise_size must be a positive integer")
        else:
            raise CustumExceptions.InvalidNoiseSizeError("You have to set the noise_size argument")

        # Handle Generator Net
        if 'generator' in kwargs:
            if kwargs['generator'] == "small_gan":
                generator = Small_GAN.GeneratorNet(noise_size=noise_size, num_classes=NUM_CLASSES,
                                                   n_image_channels=N_IMAGE_CHANNELS).to(device)
            elif kwargs['generator'] == "resNet":
                generator = ResNetGenerator.resnetGenerator(noise_size + NUM_CLASSES, N_IMAGE_CHANNELS).to(device)
            else:
                raise CustumExceptions.NoGeneratorError(
                    f'The given generator net "{kwargs["discriminator"]}" cannot be found')
        else:
            raise CustumExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")