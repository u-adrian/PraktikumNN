import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import time
import random


def train(device, data_loader, model_g, model_d, batch_size, num_epochs, noise_size, num_classes):
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # initialize One Hot encoder
    one_hot_enc = OneHotEncoder()
    all_classes = torch.tensor(range(num_classes)).reshape(-1, 1)
    one_hot_enc.fit(all_classes)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            ########
            # Update Discriminator
            ########

            # Train with all-real batch

            model_d.zero_grad()
            # Create batch (real images + desired classes)

            labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray())
            labels_one_hot = labels_one_hot[:,:,None, None]  # shape: [batch_size, num_classes, 1, 1]
            label_maps = labels_one_hot.expand(-1,-1,32,32).float()  # shape: [batch_size, num_classes, 32, 32]
            real_batch = torch.cat((images, label_maps), 1).to(device)

            # Forward pass through discriminator
            output_real = model_d(real_batch).view(-1)
            # Create labels (all true)
            real_label_batch = torch.full((batch_size, ), real_label, dtype=torch.float, device=device)
            # Calculate loss on real batch
            error_d_real = criterion(output_real, real_label_batch)
            # Backward pass through discriminator
            error_d_real.backward(retain_graph=True)

            # Train with all-fake batch

            # Create latent vectors (noise + class)
            noise = torch.randn(batch_size, noise_size, 1, 1)
            labels_one_hot = torch.tensor(one_hot_enc.transform(labels.reshape(-1, 1)).toarray())
            labels_one_hot = labels_one_hot[:, :, None, None]
            latent_vectors = torch.cat((noise, labels_one_hot), 1).to(device)
            # Generate batch (fake images + desired classes)
            fake_images = model_g(latent_vectors.float())
            fake_batch = torch.cat((fake_images, label_maps.to(device)), 1).to(device)
            # Forward pass through discriminator
            output_fake = model_d(fake_batch).view(-1)
            # Create labels (all false)
            fake_label_batch = torch.full((batch_size, ), fake_label, dtype=torch.float, device=device)
            # Calculate loss on fake batch
            error_d_fake = criterion(output_fake, fake_label_batch)
            # Backward pass through discriminator
            error_d_fake.backward(retain_graph=True)

            # Train with real-images wrong-conditions batch

            # Create batch (real images + wrong classes)
            wrong_label_maps = torch.zeros(batch_size, num_classes, 32, 32)
            for j in range(batch_size):  # TODO: solve without loop
                random_wrong_label = random.choice([x for x in range(num_classes) if x != labels[j]])
                wrong_label_maps[j, random_wrong_label] = torch.ones(32, 32)
            real_wrong_batch = torch.cat((images, wrong_label_maps), 1).to(device)
            # Forward pass through discriminator
            output_real_wrong = model_d(real_wrong_batch).view(-1)
            # Create labels (all false)
            fake_label_batch = torch.full((batch_size, ), fake_label, dtype=torch.float, device=device)
            # Calculate loss on real-images wrong-condition batch
            error_d_real_wrong = criterion(output_real_wrong, fake_label_batch)
            # Backward pass through discriminator
            error_d_real_wrong.backward(retain_graph=True)

            # Update Discriminator
            model_d.optimizer.step()

            ########
            # Update Generator
            ########

            model_g.zero_grad()
            # Forward pass through discriminator after update
            output_new_fake = model_d(fake_batch).view(-1)
            # Calculate loss on fake batch: goal is real labels (deceiving discriminator)
            error_g = criterion(output_new_fake, real_label_batch)
            # Backward pass through generator
            error_g.backward()

            # Update Generator
            model_g.optimizer.step()

            # Save statistics
            # D_x = output_real.mean().item()
            # D_G_z1 = output_fake.mean().item()
            error_d = error_d_real + error_d_fake + error_d_real_wrong
            # D_G_z2 = output_new_fake.mean().item()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'  # \tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(data_loader),
                         error_d.item(), error_g.item()))  # , D_x, D_G_z1, D_G_z2))

        # Save model every epoch
        path = 'models/gan_checkpoint'
        torch.save({
            'netG_state_dict': model_g.state_dict(),
            'netD_state_dict': model_d.state_dict(),
        }, path)

    # Save final model
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = './models/gan_' + time_stamp
    torch.save({
        'netG_state_dict': model_g.state_dict(),
        'netD_state_dict': model_d.state_dict(),
    }, path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
