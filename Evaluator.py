import json
from os import listdir
from os.path import isfile, join, basename
from pathlib import Path

import torch
import torchvision.transforms as transforms

from sklearn.preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import ArgHandler

from DataLoader import load_cifar10
from Scores import inception_score, frechet_inception_distance
from tqdm import tqdm


def evaluate_model(**kwargs):
    # Constants
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3

    # Variables
    device = ArgHandler.handle_device(**kwargs)
    noise_size = ArgHandler.handle_noise_size(**kwargs)
    generator = ArgHandler.handle_generator(NUM_CLASSES, N_IMAGE_CHANNELS, **kwargs)
    model_path = ArgHandler.handle_model_path(**kwargs)
    batch_size = ArgHandler.handle_batch_size(**kwargs)
    name = basename(model_path)
    print(f'Evaluation of model: {name}')

    # Load generator
    generator.load_state_dict(torch.load(model_path, map_location=device)['netG_state_dict'])

    # Initialize One Hot Encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(NUM_CLASSES)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    _, test_loader = load_cifar10(batch_size)
    num_images = len(test_loader.dataset)

    fakes = torch.zeros([num_images, 3, 32, 32], dtype=torch.float32)
    for i, (images, labels) in enumerate(tqdm(test_loader,desc=f'Generating {num_images} images:',leave=False), 0):
        batch_size_i = images.size()[0]
        labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)
        noise = torch.randn(batch_size_i, noise_size, 1, 1, device=device)
        with torch.no_grad():
            fake = generator(noise, labels_one_hot)
        fakes[i * batch_size:i * batch_size + batch_size_i] = fake

    print('Calculating inception score...')
    i_score = inception_score(fakes, device, batch_size)
    print('inception score: ', i_score)

    print('Calculating FID score...')
    fid_score = frechet_inception_distance(fakes, test_loader.dataset, device, batch_size)
    print('frechet inception distance: ', fid_score)

    return i_score, fid_score


def evaluate_multiple_models(**kwargs):
    model_path = ArgHandler.handle_model_path(**kwargs)

    model_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    scores_dict = dict()

    for f in model_files:
        model_kwargs = kwargs.copy()
        model_kwargs.update({"model_path": join(model_path, f)})
        try:
            i_score, fid = evaluate_model(**model_kwargs)
            scores_dict.update({f: {"inception_score": i_score, "fid": fid}})
        except Exception as e:
            print(f'An exception:\n "{e}" \n occurred for file: {f}. Will skip this')

    return scores_dict


def create_images(**kwargs):
    # Constants
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Variables
    device = ArgHandler.handle_device(**kwargs)
    noise_size = ArgHandler.handle_noise_size(**kwargs)
    generator = ArgHandler.handle_generator(NUM_CLASSES, N_IMAGE_CHANNELS, **kwargs)
    model_path = ArgHandler.handle_model_path(**kwargs)
    output_path = ArgHandler.handle_output_path(**kwargs)

    # Load generator
    generator.load_state_dict(torch.load(model_path)['netG_state_dict'])

    # Initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(NUM_CLASSES)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    # Do it multiple times
    for j in range(10):
        # Generate noise and stack 10 copies of that noise
        noise = torch.randn(1, noise_size, 1, 1, device=device)
        noise = noise.repeat(10, 1, 1, 1)

        # Create label as one hot
        labels = torch.tensor([range(10)])
        labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=device)

        # Generate batch (fake images + desired classes)
        fake_images = generator(noise, labels_one_hot)

        # As info:
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
            plt.show()

