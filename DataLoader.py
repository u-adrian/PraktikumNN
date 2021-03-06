import ssl
import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar10(batch_size, use_pseudo_augmentation=False):
    ssl._create_default_https_context = ssl._create_unverified_context  # fix download error

    if use_pseudo_augmentation:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        transforms.RandomHorizontalFlip(p=0.5)
                                        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
