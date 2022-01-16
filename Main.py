import Evaluator
import Trainer
import Experiments
import Scores
import torch


def showcase_current_project():
    device = "GPU"
    generator = "res_net_depth1"
    discriminator = "res_net_depth1"
    criterion = "BCELoss"
    learning_rate = "0.0001"
    real_img_fake_label = True
    num_epochs = "51"
    noise_size = "20"
    snapshot_interval = "1"
    batch_size = 100
    weight_init = "normal"
    augmentation = False
    name = "CurrentProject"

    # Train model
    output_path = f'./showcase/{name}'
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=output_path,
                  batch_size=batch_size, weight_init=weight_init, pseudo_augmentation=augmentation)
    # Evaluate model
    model_path = f"{output_path}/gan_latest"
    Evaluator.evaluate_model(device=device, generator=generator, noise_size=noise_size,
                             model_path=model_path, output_path=output_path, batch_size=batch_size)
    # Print images
    # Todo


def main():
    Experiments.net_configurations()
    Experiments.specialized_training()
    Experiments.specialized_training(generator="small_gan", discriminator="small_gan")
    Experiments.data_augmentation()
    Experiments.leaky_vs_normal_residual_discriminator()
    Experiments.xavier_vs_normal_init()
    Scores.inception_score_cifar10(torch.device('cuda'), batch_size=100)
    showcase_current_project()


if __name__ == "__main__":
    main()
