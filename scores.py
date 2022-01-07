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


def frechet_inception_distance(generated_images, real_images, device, batch_size=32):
    model = inception_v3(pretrained=True, transform_input=False, ).to(device)
    model.eval()



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





