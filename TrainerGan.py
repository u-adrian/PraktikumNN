import json
from os.path import join
from pathlib import Path

import torch
from sklearn.preprocessing import OneHotEncoder

from torch import optim

import ArgHandler
import DataLoader
from tqdm import tqdm


def train(**kwargs):
    t = Trainer(**kwargs)
    t.train()


class Trainer:
    # Constant
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3
    betas = (0.5, 0.999)

    def __init__(self, **kwargs):
        self._parse_args(**kwargs)
        self._create_folder_structure()
        kwargs_copy = kwargs.copy()
        with open(join(self.output_path, 'Arguments.txt'), "w+") as arguments_file:
            arguments_file.write(json.dumps(kwargs_copy))

    def train(self):
        train_loader, _ = DataLoader.load_cifar10(self.batch_size, use_pseudo_augmentation=self.data_augmentation)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Initialize One Hot Encoder
        one_hot_enc = OneHotEncoder()
        all_classes = torch.tensor(range(self.NUM_CLASSES)).reshape(-1, 1)
        one_hot_enc.fit(all_classes)

        # Training Loop
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader,desc=f'Epoch {epoch}/{self.num_epochs}',leave=False), 0):
                ############################
                # (1) Update Discriminator network
                ###########################
                # Train with all-real batch
                batch_size_i = images.size()[0]
                self.discriminator.zero_grad()
                real_images = images.to(self.device)
                labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(), device=self.device)
                real_labels = torch.full((batch_size_i,), real_label, dtype=torch.float, device=self.device)
                output = self.discriminator(real_images, labels_one_hot.detach()).view(-1)
                errD_real = self.criterion(output, real_labels)
                errD_real.backward(retain_graph=True)

                # Train with all-fake batch
                noise = torch.randn(batch_size_i, self.noise_size, 1, 1, device=self.device)
                fake = self.generator(noise, labels_one_hot)
                fake_labels = torch.full((batch_size_i,), fake_label, dtype=torch.float, device=self.device)
                output = self.discriminator(fake.detach(), labels_one_hot.detach()).view(-1)
                errD_fake = self.criterion(output, fake_labels)
                errD_fake.backward(retain_graph=True)

                # Train with real images and fake labels (rifl)
                if self.real_img_fake_label:
                    deviation_labels = labels + torch.randint(low=1, high=10, size=labels.shape)
                    deviation_labels = torch.remainder(deviation_labels, 10)
                    deviation_one_hot = torch.tensor(one_hot_enc.transform(deviation_labels.reshape(-1, 1)).toarray(),
                                                     device=self.device)

                    output = self.discriminator(real_images, deviation_one_hot).view(-1)
                    errD_fake_labels = self.criterion(output, fake_labels)
                    errD_fake_labels.backward(retain_graph=True)

                # Update the discriminator net
                self.discriminator.optimizer.step()

                ############################
                # (2) Update Generator network
                ###########################
                self.generator.zero_grad()
                output = self.discriminator(fake, labels_one_hot).view(-1)
                errG = self.criterion(output, real_labels)
                errG.backward()
                self.generator.optimizer.step()

            # Save snapshot of the model
            if self.do_snapshots and self.snapshot_interval > 0 and epoch % self.snapshot_interval == 0:
                path = f'{self.output_path}/snapshots/gan_after_epoch_{epoch}'
                torch.save({
                    'netG_state_dict': self.generator.state_dict(),
                    'netD_state_dict': self.discriminator.state_dict(),
                }, path)

        # Save model
        path = f'{self.output_path}/gan_latest'
        torch.save({
            'netG_state_dict': self.generator.state_dict(),
            'netD_state_dict': self.discriminator.state_dict(),
        }, path)

    def _create_folder_structure(self):
        Path(f'{self.output_path}/snapshots/').mkdir(parents=True, exist_ok=True)

    def _parse_args(self, **kwargs):
        # Handle Arguments
        self.device = ArgHandler.handle_device(**kwargs)

        self.learning_rate = ArgHandler.handle_learning_rate(**kwargs)

        self.noise_size = ArgHandler.handle_noise_size(**kwargs)

        self.generator = ArgHandler.handle_generator(self.NUM_CLASSES, self.N_IMAGE_CHANNELS, **kwargs)

        if ArgHandler.handle_pretrained_generator(**kwargs):
            model_path = ArgHandler.handle_model_path(**kwargs)
            print('Loading generator net...')
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device)['netG_state_dict'])
        else:
            self.generator.apply(ArgHandler.handle_weights_init(**kwargs))

        self.generator.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                              betas=self.betas)

        self.discriminator = ArgHandler.handle_discriminator(self.NUM_CLASSES, self.N_IMAGE_CHANNELS, **kwargs)

        self.discriminator.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                                  betas=self.betas)

        self.discriminator.apply(ArgHandler.handle_weights_init(**kwargs))

        self.num_epochs = ArgHandler.handle_num_epochs(**kwargs)

        self.batch_size = ArgHandler.handle_batch_size(**kwargs)

        self.criterion = ArgHandler.handle_criterion(**kwargs)

        self.real_img_fake_label = ArgHandler.handle_real_img_fake_label(**kwargs)

        self.snapshot_interval, self.do_snapshots = ArgHandler.handle_snapshot_settings(**kwargs)

        self.output_path = ArgHandler.handle_output_path(**kwargs)

        self.data_augmentation = ArgHandler.handle_pseudo_augment(**kwargs)
