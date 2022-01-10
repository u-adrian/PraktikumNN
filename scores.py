import torch
from torchvision.models.inception import inception_v3
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import ssl
import CustomExceptions
from torchsummary import summary
import self
from scipy import linalg

def load_cifar10_for_inception(batch_size):
    # fix download error
    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(299),
                                    transforms.CenterCrop(299),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader



# To test if inception score is implemented correctly
def inception_score_cifar10(device, batch_size=32):
    if device == "GPU":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise CustomExceptions.GpuNotFoundError("Cannot find a CUDA device")
    else:
        device = torch.device('cpu')

    dataloader = load_cifar10_for_inception(batch_size)
    num_images = len(dataloader.dataset)

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    predictions = np.zeros((num_images, 1000))

    for i, (batch, labels) in enumerate(dataloader, 0):
        print('Predicting labels with inception v3 model: ', i * batch_size, '/', num_images)
        batch = batch.to(device)
        batch_size_i = batch.size()[0]

        with torch.no_grad():
            prediction = model(batch)

        prediction = F.softmax(prediction, dim=1).data.cpu().numpy()

        predictions[i * batch_size:i * batch_size + batch_size_i] = prediction

    print('Calculating inception score...')
    py = np.mean(predictions, axis=0)
    scores = []

    for i in range(num_images):
        pyx = predictions[i, :]
        scores.append(entropy(pyx, py))

    print('inception score: ', np.exp(np.mean(scores)))
    return


#
# Inception score measures how realistic a GAN's output is:
# It measures if the images have variety and if each images looks distinctly like something.
# If both things are true then the inception score is high,
# if either one of the measures are false the inception score is low.
def inception_score(images, device, batch_size=32):

    num_images = images.shape[0]

    dataset = FakeDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    predictions = np.zeros((num_images, 1000))

    for i, batch in enumerate(dataloader, 0):
        print('Predicting labels with inception v3 model: ', i * batch_size, '/', num_images)
        batch = batch.to(device)
        batch_size_i = batch.size()[0]

        with torch.no_grad():
            prediction = model(batch)

        prediction = F.softmax(prediction, dim=1).data.cpu().numpy()

        predictions[i * batch_size:i * batch_size + batch_size_i] = prediction


    print('Calculating inception score...')
    py = np.mean(predictions, axis=0)
    scores = []

    for i in range(num_images):
        pyx = predictions[i, :]
        scores.append(entropy(pyx, py))

    return np.exp(np.mean(scores))


# A lower FID indicates better-quality images abd a higher score indicates a lower-quality image.
def frechet_inception_distance(generated_images, real_dataset, device, batch_size=32, eps=1e-6):
    generated_dataset = FakeDataset(generated_images)
    generated_dataloader = DataLoader(generated_dataset, batch_size)

    real_dataset.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    real_dataloader = DataLoader(real_dataset, batch_size)

    assert len(generated_dataset) == len(real_dataset)
    num_images = len(generated_dataset)

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.avgpool.register_forward_hook(get_activation('avgpool'))

    real_activations = np.zeros((num_images, 2048))

    for i, (real_image_batch, _) in enumerate(real_dataloader, 0):
        print('Calculating activations of avgpool layer of inception v3 model: ', i * batch_size, '/', num_images)
        batch_size_i = real_image_batch.size()[0]

        real_image_batch.to(device)
        with torch.no_grad():
            output = model(real_image_batch)
        real_activations[i * batch_size:i * batch_size + batch_size_i] = activation['avgpool'].squeeze(3).squeeze(2).data.cpu().numpy()


    generated_activations = np.zeros((num_images, 2048))

    for i, generated_image_batch in enumerate(generated_dataloader):
        print('Calculating activations of generated images: ', i * batch_size, '/', num_images)
        batch_size_i = generated_image_batch.size()[0]
        generated_image_batch.to(device)
        with torch.no_grad():
            output = model(generated_image_batch)
        generated_activations[i * batch_size: i * batch_size + batch_size_i] = activation['avgpool'].squeeze(3).squeeze(2).data.cpu().numpy()

    print('Calculating FID score...')

    real_mean = np.mean(real_activations, axis=0)
    real_sigma = np.cov(real_activations, rowvar=False)

    generated_mean = np.mean(generated_activations, axis=0)
    generated_sigma = np.cov(generated_activations, rowvar=False)

    ssdiff = np.sum((generated_mean - real_mean)**2.0)
    covmean = linalg.sqrtm(generated_sigma.dot(real_sigma))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(generated_sigma + real_sigma - 2.0 * covmean)
    return fid


class FakeDataset(Dataset):
    def __init__(self, fake_images):
        self.fake_images = fake_images
        self.transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, item):
        fake_image = self.fake_images[item]
        fake_image_scaled = self.transforms(fake_image)
        return fake_image_scaled

    def __len__(self):
        return self.fake_images.shape[0]






