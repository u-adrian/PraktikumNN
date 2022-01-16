import time

import torch
from torch import nn

import CustomExceptions
from Nets import Utils
from Nets.ResNet import ResNetGenerator, ResNetDiscriminator
from Nets.SmallGan import Small_GAN


def handle_pretrain(**kwargs):
    pretrain = False
    if 'pretrain' in kwargs:
        try:
            pretrain = bool(kwargs['pretrain'])
        except ValueError:
            raise CustomExceptions.InvalidArgumentError('pretrain must be bool')
    return pretrain


def handle_pretrained_generator(**kwargs):
    pretrained_generator = False
    if 'pretrained_generator' in kwargs:
        try:
            pretrained_generator = bool(kwargs['pretrained_generator'])
        except ValueError:
            raise CustomExceptions.InvalidArgumentError('pretained_generator must be bool')
    return pretrained_generator


def handle_pretrained_encoder(**kwargs):
    pretrained_encoder = False
    if 'pretrained_encoder' in kwargs:
        try:
            pretrained_encoder = bool(kwargs['encoder_generator'])
        except ValueError:
            raise CustomExceptions.InvalidArgumentError('pretained_encoder must be bool')
    return pretrained_encoder


def handle_noise_size(**kwargs):
    noise_size = -1
    if 'noise_size' in kwargs:
        try:
            noise_size = int(kwargs['noise_size'])
            if noise_size <= 0:
                raise CustomExceptions.InvalidNoiseSizeError("noise_size must be greater than 0.")
        except ValueError:
            raise CustomExceptions.InvalidNoiseSizeError("noise_size must be a positive integer")
    else:
        raise CustomExceptions.InvalidNoiseSizeError("You have to set the noise_size argument")
    return noise_size


def handle_num_epochs(**kwargs):
    num_epochs = -1
    if 'num_epochs' in kwargs:
        try:
            num_epochs = int(kwargs['num_epochs'])
            if num_epochs <= 0:
                raise CustomExceptions.NumEpochsError("The Number of epochs must be greater than 0")
        except ValueError:
            raise CustomExceptions.NumEpochsError("The Number of epochs must be a positive integer")
    else:
        raise CustomExceptions.NumEpochsError("The Number of epochs must be defined")
    return num_epochs


def handle_batch_size(**kwargs):
    batch_size = -1
    if 'batch_size' in kwargs:
        try:
            batch_size = int(kwargs['batch_size'])
            if batch_size <= 0:
                raise CustomExceptions.BatchSizeError("The batch size must be greater than 0!")
        except ValueError:
            raise CustomExceptions.BatchSizeError("The batch size must be a positive integer")
    else:
        raise CustomExceptions.BatchSizeError("The batch size must be defined")
    return batch_size


def handle_learning_rate(**kwargs):
    if 'learning_rate' in kwargs:
        try:
            learning_rate = float(kwargs['learning_rate'])
        except ValueError:
            raise CustomExceptions.LearningRateError("The learning rate must be float")
    else:
        raise CustomExceptions.LearningRateError("The learning rate must be defined. Use 'learning_rate=0.0001' for example")
    return learning_rate


def handle_criterion(**kwargs):
    criterion = None
    if 'criterion' in kwargs:
        if kwargs['criterion'] == 'BCELoss':
            criterion = nn.BCELoss()
        elif kwargs['criterion'] == 'Wasserstein':
            criterion = 'Wasserstein'  # TODO
            raise NotImplementedError
        elif kwargs['criterion'] == 'MiniMax':
            criterion = 'MiniMax'  # TODO
            raise NotImplementedError
        else:
            raise CustomExceptions.InvalidLossError()
    else:
        raise CustomExceptions.InvalidLossError()
    return criterion


def handle_real_img_fake_label(**kwargs):
    real_img_fake_label = False
    if 'real_img_fake_label' in kwargs:
        if type(kwargs['real_img_fake_label']) is bool:
            return kwargs['real_img_fake_label']
        if kwargs['real_img_fake_label'].lower() in ['true', 't', 'yes', 'y', '1']:
            real_img_fake_label = True
        elif kwargs['real_img_fake_label'].lower() in ['false', 'f', 'no', 'n', '0']:
            real_img_fake_label = False
        else:
            raise CustomExceptions.InvalidArgumentError(
                f'Invalid argument for "real_img_fake_label": "{kwargs["real_img_fake_label"]}"')
    else:
        print(f'No argument for "real_img_fake_label" found. Will use {real_img_fake_label}')
    return real_img_fake_label


def handle_snapshot_settings(**kwargs):
    snapshot_interval = -1
    do_snapshots = False
    if 'snapshot_interval' in kwargs:
        try:
            snapshot_interval = int(kwargs['snapshot_interval'])
            do_snapshots = True
            if snapshot_interval <= 0:
                raise CustomExceptions.InvalidSnapshotInterval("snapshot_interval must be greater than 0.")
        except ValueError:
            raise CustomExceptions.InvalidNoiseSizeError("snapshot_interval must be a positive integer")
    else:
        do_snapshots = False
        print(f'snapshot_interval is not defined. Will not create nor store snapshots')
    return snapshot_interval, do_snapshots


def handle_name(**kwargs):
    name = '#'
    unique_name = '#'
    if 'name' in kwargs:
        name = kwargs['name']
        unique_name = f'{name}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        if not name.isalnum():
            raise CustomExceptions.InvalidNameError("name must only contain alphanumeric characters")
    else:
        unique_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        print("No name given. Will only use the time-stamp")
    return name, unique_name



def handle_device(**kwargs):
    device = None
    if 'device' in kwargs:
        if kwargs['device'] == "GPU":
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise CustomExceptions.GpuNotFoundError("Cannot find a CUDA device")
        else:
            device = torch.device('cpu')
    return device


def handle_generator(NUM_CLASSES, N_IMAGE_CHANNELS, **kwargs):
    device = handle_device(**kwargs)
    noise_size = handle_noise_size(**kwargs)
    generator = None
    if 'generator' in kwargs:
        if kwargs['generator'] == "small_gan":
            generator = Small_GAN.GeneratorNet(noise_size=noise_size, num_classes=NUM_CLASSES,
                                               n_image_channels=N_IMAGE_CHANNELS).to(device)
        elif kwargs['generator'] == "res_net_depth1":
            generator = ResNetGenerator.resnetGeneratorDepth1(noise_size + NUM_CLASSES, N_IMAGE_CHANNELS).to(device)
        elif kwargs['generator'] == "res_net_depth2":
            generator = ResNetGenerator.resnetGeneratorDepth2(noise_size + NUM_CLASSES, N_IMAGE_CHANNELS).to(device)
        else:
            raise CustomExceptions.NoGeneratorError(
                f'The given generator net "{kwargs["generator"]}" cannot be found')
    else:
        raise CustomExceptions.NoGeneratorError("You need to define the generator net. keyword: 'generator'")
    return generator


def handle_discriminator(NUM_CLASSES, N_IMAGE_CHANNELS, **kwargs):
    device = handle_device(**kwargs)
    discriminator = None
    if 'discriminator' in kwargs:
        if kwargs['discriminator'] == "small_gan":
            discriminator = Small_GAN.DiscriminatorNet(n_image_channels=N_IMAGE_CHANNELS, num_classes=NUM_CLASSES).to(device)
        elif kwargs['discriminator'] == "res_net_depth1":
            discriminator = ResNetDiscriminator.resnetDiscriminatorDepth1(N_IMAGE_CHANNELS + NUM_CLASSES, 1).to(device)
        elif kwargs['discriminator'] == "res_net_depth1_leaky":
            discriminator = ResNetDiscriminator.resnetDiscriminatorDepth1Leaky(N_IMAGE_CHANNELS + NUM_CLASSES, 1).to(device)
        elif kwargs['discriminator'] == "res_net_depth2":
            discriminator = ResNetDiscriminator.resnetDiscriminatorDepth2(N_IMAGE_CHANNELS + NUM_CLASSES, 1).to(device)
        else:
            raise CustomExceptions.NoDiscriminatorError(
                f'The given discriminator net "{kwargs["discriminator"]}" cannot be found')
    else:
        raise CustomExceptions.NoDiscriminatorError(
            "You need to define the discriminator net. keyword: 'discriminator'")
    return discriminator


def handle_model_path(**kwargs):
    if 'model_path' in kwargs:
        return kwargs['model_path']
    else:
        raise NotImplementedError('Using a default model_path is not implemented. And will never be')

def handle_output_path(**kwargs):
    if 'output_path' in kwargs:
        return kwargs['output_path']
    else:
        raise NotImplementedError('Using a default output_path is not implemented. And will never be')


def handle_weights_init(**kwargs):
    if 'weights_init' in kwargs:
        if kwargs['weights_init'] == "normal":
            return Utils.weights_init
        elif kwargs['weights_init'] == "xavier":
            return Utils.weights_init_xavier
        else:  # default
            print("Default normal weight init is used")
            return Utils.weights_init
    else:  # default
        print("Default normal weight init is used")
        return Utils.weights_init


def handle_pseudo_augment(**kwargs):
    if 'pseudo_augment' in kwargs:
        if type(kwargs['pseudo_augment']) is bool:
            return kwargs['pseudo_augment']
        if kwargs['pseudo_augment'].lower() in ['true', 't', 'yes', 'y', '1']:
            return True
        elif kwargs['pseudo_augment'].lower() in ['false', 'f', 'no', 'n', '0']:
            return False
    else:
        print("Won't use data augmentation.")
        return False
