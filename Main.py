import Experiments
import Scores

import torch


def main():
    Experiments.experiment_leaky_vs_normal()
    # Experiments.experiment_xavier_vs_normal()
    # Experiments.scores_stability()
    # Experiments.data_augmentation_experiment()
    # Scores.inception_score_cifar10(torch.device('cuda'), batch_size=100)


if __name__ == "__main__":
    main()
