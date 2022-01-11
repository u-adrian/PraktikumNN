import Evaluator
import Trainer
from scores import inception_score_cifar10


def experiment_leaky_vs_normal():
    # Parameters
    device = "GPU"
    generator = "res_net_depth1"
    discriminator = "res_net_depth1_leaky"
    criterion = "BCELoss"
    learning_rate = "0.0001"
    real_img_fake_label = "False"
    num_epochs = "51"
    noise_size = "20"
    snapshot_interval = "5"
    name = "resNet01Leaky"
    batch_size = 100

    # Train model with Leaky ReLu
    unique_name = Trainer.train(device, generator, discriminator, criterion, learning_rate, real_img_fake_label,
                                num_epochs, noise_size, snapshot_interval, name, batch_size)
    scores_path = f'./models/{unique_name}/snapshots'
    Evaluator.evaluate_multiple_models(device, generator, noise_size, scores_path, scores_path, batch_size)

    # Changes
    discriminator = "res_net_depth1"
    name = "resNet01NonLeaky"

    # Train model with normal ReLu
    unique_name = Trainer.train(device, generator, discriminator, criterion, learning_rate, real_img_fake_label,
                                num_epochs, noise_size, snapshot_interval, name, batch_size)
    scores_path = f'./models/{unique_name}/snapshots'
    Evaluator.evaluate_multiple_models(device, generator, noise_size, scores_path, scores_path, batch_size)


def main():
    experiment_leaky_vs_normal()


if __name__ == "__main__":
    main()
