import json
from os.path import join

import Evaluator
import Trainer
import Experiments
import Scores
import torch
from pathlib import Path


def showcase_current_project():
    device = "GPU"
    generator = "res_net_depth1"
    discriminator = "res_net_depth1"
    criterion = "BCELoss"
    learning_rate = 0.0001
    real_img_fake_label = True
    num_epochs = 31
    noise_size = 20
    snapshot_interval = 5
    batch_size = 100
    weight_init = "normal"
    augmentation = False
    name = "CurrentProject"

    # Train model
    print(f"Started training of model: {name}")
    output_path = f'./showcase/{name}'
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=output_path,
                  batch_size=batch_size, weight_init=weight_init, pseudo_augmentation=augmentation)
    print(f"Finished training of model: {name}")

    # Evaluate model
    print(f"Started evaluation of model: {name}")
    model_path = f"{output_path}/gan_latest"
    scores_dict = Evaluator.evaluate_model(device=device, generator=generator, noise_size=noise_size,
                                           model_path=model_path, output_path=output_path, batch_size=batch_size)
    print(f"Finished evaluation of model: {name}")

    # Save scores
    with open(join(output_path, 'scores.txt'), "w+") as scores_file:
        scores_file.write(json.dumps(scores_dict))
    print(f"Stored scores")

    # Print images
    Evaluator.create_images(device=device, generator=generator, noise_size=noise_size, model_path=model_path,
                            output_path=output_path)


def main():
    # Experiments.net_configurations()
    # Experiments.specialized_training()
    # Experiments.specialized_training(generator="small_gan", discriminator="small_gan")
    # Experiments.data_augmentation()
    # Experiments.leaky_vs_normal_residual_discriminator()
    # Experiments.xavier_vs_normal_init()
    # Experiments.generator_pretraining()
    Experiments.generator_pretraining(num_epochs=5)
    # Scores.inception_score_cifar10(torch.device('cuda'), batch_size=100)
    # showcase_current_project()


if __name__ == "__main__":
    main()
