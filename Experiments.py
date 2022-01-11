import Evaluator
import Trainer
from scores import inception_score_cifar10


def experiment_leaky_vs_normal():
    # Parameters
    device = "GPU"
    generator = "res_net_depth1"
    criterion = "BCELoss"
    learning_rate = "0.0001"
    real_img_fake_label = "False"
    num_epochs = "51"
    noise_size = "20"
    snapshot_interval = "5"
    batch_size = 100
    # Important
    name = "resNet01Leaky"
    discriminator = "res_net_depth1_leaky"

    # Train model with Leaky ReLu
    unique_name = Trainer.train(device=device, generator=generator, discriminator=discriminator,
                                criterion=criterion, learning_rate=learning_rate,
                                real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                                snapshot_interval=snapshot_interval, name=name, batch_size=batch_size)
    scores_path = f'./models/{unique_name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)

    # Changes
    name = "resNet01NonLeaky"
    discriminator = "res_net_depth1"

    # Train model with normal ReLu
    unique_name = Trainer.train(device=device, generator=generator, discriminator=discriminator,
                                criterion=criterion, learning_rate=learning_rate,
                                real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                                snapshot_interval=snapshot_interval, name=name, batch_size=batch_size)
    scores_path = f'./models/{unique_name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)


def experiment_xavier_vs_normal():
    # Parameters
    device = "GPU"
    generator = "res_net_depth1"
    discriminator = "res_net_depth1"
    criterion = "BCELoss"
    learning_rate = "0.0001"
    real_img_fake_label = "False"
    num_epochs = "51"
    noise_size = "20"
    snapshot_interval = "5"
    batch_size = 100
    # Important
    name = "resNet01XavierWeightInit"
    weight_init = "xavier"

    # Train model with Leaky ReLu
    unique_name = Trainer.train(device=device, generator=generator, discriminator=discriminator,
                                criterion=criterion, learning_rate=learning_rate,
                                real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                                snapshot_interval=snapshot_interval, name=name, batch_size=batch_size,
                                weight_init=weight_init)
    scores_path = f'./models/{unique_name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)

    # Changes
    name = "resNet01NormalWeightInit"
    weight_init = "normal"

    # Train model with normal ReLu
    unique_name = Trainer.train(device=device, generator=generator, discriminator=discriminator,
                                criterion=criterion, learning_rate=learning_rate,
                                real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                                snapshot_interval=snapshot_interval, name=name, batch_size=batch_size,
                                weight_init=weight_init)
    scores_path = f'./models/{unique_name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)


def main():
    experiment_leaky_vs_normal()
    experiment_xavier_vs_normal()


if __name__ == "__main__":
    main()
