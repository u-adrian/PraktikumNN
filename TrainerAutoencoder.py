from pathlib import Path

import torch
from sklearn.preprocessing import OneHotEncoder

from torch import optim
from torch import nn

import ArgHandler
import Data_Loader
from tqdm import tqdm

from Nets.Autoencoder.EncoderNet import EncoderNet


def train(**kwargs):
    t = TrainerAutoencoder(**kwargs)
    t.train()


class TrainerAutoencoder:
    # CONSTANT
    NUM_CLASSES = 10
    N_IMAGE_CHANNELS = 3
    criterion = nn.MSELoss()

    # VARIABLES
    output_path = None

    do_snapshots = False
    snapshot_interval = None

    device = None
    generator = None
    num_epochs = None
    batch_size = None
    noise_size = None
    learning_rate = None
    data_augmentation = None
    betas = (0.5, 0.999)  # TODO

    pretrained_generator = False
    pretrained_encoder = False
    pretrain = False

    def __init__(self, **kwargs):
        self._parse_args(**kwargs)
        self._create_folder_structure()

    def train(self):
        train_loader, _ = Data_Loader.load_cifar10(self.batch_size, use_pseudo_augmentation=self.data_augmentation)

        # initialize One Hot encoder
        one_hot_enc = OneHotEncoder()
        all_classes = torch.tensor(range(self.NUM_CLASSES)).reshape(-1, 1)
        one_hot_enc.fit(all_classes)


        # Training Loop
        for epoch in range(self.num_epochs):
            for i, (real_images, labels) in enumerate(
                    tqdm(train_loader, desc=f'Epoch {epoch}/{self.num_epochs}', leave=False), 0):
                ############################
                # Update Encoder and Generator(Decoder) network
                ############################

                labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray(),
                                              device=self.device)

                self.generator.zero_grad()
                self.encoder.zero_grad()

                gpu_data = real_images.to(self.device)

                generated_images = self.generator(self.encoder(gpu_data), labels_one_hot)

                loss = self.criterion(generated_images, gpu_data)
                loss.backward()

                self.generator.optimizer.step()
                self.encoder.optimizer.step()

            if self.do_snapshots and self.snapshot_interval > 0 and epoch % self.snapshot_interval == 0:
                path = f'{self.output_path}/snapshots/gan_after_epoch_{epoch}'
                torch.save({
                    'netG_state_dict': self.generator.state_dict(),
                    'netE_state_dict': self.encoder.state_dict(),
                }, path)

        # Save model
        path = f'{self.output_path}/gan_latest'
        torch.save({
            'netG_state_dict': self.generator.state_dict(),
            'netE_state_dict': self.encoder.state_dict(),
        }, path)

    def _create_folder_structure(self):
        Path(f'{self.output_path}/snapshots/').mkdir(parents=True, exist_ok=True)

    def _parse_args(self, **kwargs):
        # Handle Arguments
        self.device = ArgHandler.handle_device(**kwargs)

        self.learning_rate = ArgHandler.handle_learning_rate(**kwargs)

        self.noise_size = ArgHandler.handle_noise_size(**kwargs)

        self.generator = ArgHandler.handle_generator(self.NUM_CLASSES, self.N_IMAGE_CHANNELS, **kwargs)

        self.pretrained_generator = ArgHandler.handle_pretrained_generator(**kwargs)

        if self.pretrained_generator:
            model_path = ArgHandler.handle_model_path(**kwargs)
            print('Loading generator net...')
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device)['netG_state_dict'])
        else:
            self.generator.apply(ArgHandler.handle_weights_init(**kwargs))

        self.generator.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                              betas=self.betas)

        self.encoder = EncoderNet(self.N_IMAGE_CHANNELS, self.noise_size).to(self.device)

        self.pretrained_encoder = ArgHandler.handle_pretrained_encoder(**kwargs)

        if not self.pretrained_encoder:
            self.encoder.apply(ArgHandler.handle_weights_init(**kwargs))
        else:
            model_path = ArgHandler.handle_model_path(**kwargs)
            print('Loading generator net...')
            self.encoder.load_state_dict(torch.load(model_path, map_location=self.device)['netE_state_dict'])

        self.encoder.optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                            betas=self.betas)

        self.num_epochs = ArgHandler.handle_num_epochs(**kwargs)

        self.batch_size = ArgHandler.handle_batch_size(**kwargs)

        self.snapshot_interval, self.do_snapshots = ArgHandler.handle_snapshot_settings(**kwargs)

        self.output_path = ArgHandler.handle_output_path(**kwargs)

        self.data_augmentation = ArgHandler.handle_pseudo_augment(**kwargs)
