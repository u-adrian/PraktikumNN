import json
from os.path import join
from pathlib import Path

import TrainerAutoencoder
import Evaluator
import TrainerGan


def net_configurations(experiment_path=f'./experiments/net_configs',
                       device="GPU",
                       criterion="BCELoss",
                       learning_rate=0.0001,
                       real_img_fake_label=True,
                       num_epochs=51,
                       noise_size=20,
                       snapshot_interval=10,
                       batch_size=100,
                       weights_init="normal",
                       augmentation=False,
                       pretraining=False,
                       model_path=None):
    """
    This experiment trains and evaluates different GAN architectures
    """
    # Parameters for experiment
    options = [("SmallGan", "small_gan", "small_gan"),
               ("ResGanDepth1", "res_net_depth1", "res_net_depth1"),
               ("ResGanDepth2", "res_net_depth2", "res_net_depth2")]
    # Run experiments
    for name, generator, discriminator in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weights_init, augmentation, pretraining, model_path)


def specialized_training(experiment_path=f'./experiments/rifl_training',
                         device="GPU",
                         generator="res_net_depth1",
                         discriminator="res_net_depth1",
                         criterion="BCELoss",
                         learning_rate=0.0001,
                         num_epochs=51,
                         noise_size=20,
                         snapshot_interval=10,
                         batch_size=100,
                         weights_init="normal",
                         augmentation=False,
                         pretraining=False,
                         model_path=None):
    """
    This experiment trains and evaluates a GAN with and without special training on real images and false labels
    """
    # Parameters for experiment
    options = [("WithRifl", True),
               ("WithoutRifl", False)]
    # Run experiments
    for name, real_img_fake_label in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weights_init, augmentation, pretraining, model_path)


def leaky_vs_normal_residual_discriminator(experiment_path=f'./experiments/leaky_vs_normal',
                                           device="GPU",
                                           generator="res_net_depth1",
                                           criterion="BCELoss",
                                           learning_rate=0.0001,
                                           real_img_fake_label=True,
                                           num_epochs=51,
                                           noise_size=20,
                                           snapshot_interval=10,
                                           batch_size=100,
                                           weights_init="normal",
                                           augmentation=False,
                                           pretraining=False,
                                           model_path=None):
    """
    This experiment trains and evaluates a GAN with leaky RelU and with normal ReLU as activation in the Discriminator
    """
    # Parameters for experiment
    options = [("LeakyResDiscriminator", "res_net_depth1_leaky"),
               ("ReluResDiscriminator", "res_net_depth1")]
    # Run experiments
    for name, discriminator in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weights_init, augmentation, pretraining, model_path)


def xavier_vs_normal_init(experiment_path=f'./experiments/xavier_vs_normal',
                          device="GPU",
                          generator="res_net_depth1",
                          discriminator="res_net_depth1",
                          criterion="BCELoss",
                          learning_rate=0.0001,
                          real_img_fake_label=True,
                          num_epochs=51,
                          noise_size=20,
                          snapshot_interval=10,
                          batch_size=100,
                          augmentation=False,
                          pretraining=False,
                          model_path=None):
    """
    This experiment trains and evaluates a GAN with xavier and with normal weights initialization
    """
    # Parameters for experiment
    options = [("XavierInit", "xavier"),
               ("NormalInit", "normal")]
    # Run experiments
    for name, weights_init in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weights_init, augmentation, pretraining, model_path)


def data_augmentation(experiment_path=f'./experiments/data_aug',
                      device="GPU",
                      generator="small_gan",
                      discriminator="small_gan",
                      criterion="BCELoss",
                      learning_rate=0.0001,
                      real_img_fake_label=True,
                      num_epochs=101,
                      noise_size=20,
                      snapshot_interval=10,
                      batch_size=100,
                      weights_init="normal",
                      pretraining=False,
                      model_path=None):
    """
    This experiment trains and evaluates a GAN with and without augmentation of the training data
    """
    # Parameters for experiment
    options = [("WithoutAugmentation", False),
               ("WithAugmentation", True)]
    # Run experiments
    for name, augmentation in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weights_init, augmentation, pretraining, model_path)


def generator_pretraining(experiment_path=f'./experiments/pretraining',
                          device="GPU",
                          generator="small_gan",
                          discriminator="small_gan",
                          criterion="BCELoss",
                          learning_rate=0.0001,
                          real_img_fake_label=True,
                          num_epochs=51,
                          noise_size=20,
                          snapshot_interval=10,
                          batch_size=100,
                          weights_init="normal",
                          augmentation=False,
                          num_epochs_pretraining=10):
    """
    This experiment trains and evaluates a GAN with and without pretraining of the training generator
    """
    # Parameters for experiment
    options = [("WithPretraining", True),
               ("WithoutPretraining", False)]
    # Run experiments
    for name, pretraining in options:
        if pretraining:
            # Pretrain generator as autoencoder
            print(f"Started pretraining of generator")
            model_path = f'./{experiment_path}/models/{name}'
            TrainerAutoencoder.train(device=device, generator=generator, learning_rate=learning_rate,
                                     num_epochs=num_epochs_pretraining, noise_size=noise_size,
                                     snapshot_interval=snapshot_interval, output_path=model_path,
                                     batch_size=batch_size, weights_init=weights_init, augmentation=augmentation)
            print(f"Finished pretraining of generator")
        else:
            model_path = None
        # Normal training and evaluation
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weights_init, augmentation, pretraining, model_path)


def _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                        real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                        weights_init, augmentation, pretraining, model_path):
    """
    This method trains and evaluates a GAN with the given parameters
    Args:
        experiment_path: directory where the results of training and evaluation are stored
        name: name of the model to be trained
        device: device on which the training is executed. either GPU or CPU.
        generator: specifier of the generator net
        discriminator: specifier of the discriminator net
        criterion: criterion used to calculate the loss
        learning_rate: learning rate for the training
        real_img_fake_label: whether special training on real images and false labels should be used
        num_epochs: number of epochs for the training
        noise_size: size of the noise used in the generator
        snapshot_interval: number of epochs between saving a snapshot of the training
        batch_size: size of the batch for the training
        weights_init: weights initialization used for the generator and discriminator
        augmentation: whether augmentation should be used  for the training data
        pretraining: whether the training should load a pretrained generator from model_path
        model_path: path to a pretrained generator
    """
    # Train model
    print(f"Started training of model: {name}")
    output_path = f'./{experiment_path}/models/{name}'
    TrainerGan.train(device=device, generator=generator, discriminator=discriminator,
                     criterion=criterion, learning_rate=learning_rate,
                     real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                     snapshot_interval=snapshot_interval, output_path=output_path,
                     batch_size=batch_size, weights_init=weights_init, pseudo_augmentation=augmentation,
                     pretraining=pretraining, model_path=model_path)
    print(f"Finished training of model: {name}")

    # Evaluate snapshots
    print(f"Started evaluation of model: {name}")
    snapshot_path = f'./{experiment_path}/models/{name}/snapshots'
    scores_dict = Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                                     model_path=snapshot_path, batch_size=batch_size)
    # Save scores
    Path(f'{snapshot_path}/').mkdir(parents=True, exist_ok=True)
    with open(join(snapshot_path, 'scores.txt'), "w+") as scores_file:
        scores_file.write(json.dumps(scores_dict))
    print(f"Stored scores")

    # Evaluate model
    latest_model = f'./{experiment_path}/models/{name}/gan_latest'
    scores_dict = Evaluator.evaluate_model(device=device, generator=generator, noise_size=noise_size,
                                           model_path=latest_model, batch_size=batch_size)
    print(f"Finished evaluation of model: {name}")

    # Save scores
    Path(f'{output_path}/').mkdir(parents=True, exist_ok=True)
    with open(join(output_path, 'scores.txt'), "w+") as scores_file:
        scores_file.write(json.dumps(scores_dict))
    print(f"Stored scores")


#####################
# Special Experiment to Run on BW Cluster
#####################


def sig_opt_experiment(**kwargs):
    # Parameters
    device = "GPU"  # constant
    criterion = "BCELoss"  # constant
    batch_size = 100  # constant
    model_output_path = kwargs['model_output_path']

    model_args = kwargs['suggestion']
    # 6 params in kwargs
    generator = model_args['generator_and_discriminator']
    if generator == "res_net_depth1_leaky":
        generator = "res_net_depth1"

    discriminator = model_args['generator_and_discriminator']

    learning_rate = model_args['learning_rate']

    rifl_bool = model_args['real_img_fake_label']
    real_img_fake_label = f'{rifl_bool}'

    num_epochs = model_args['num_epochs']
    noise_size = model_args['noise_size']

    weights_init = model_args['weights_init']

    # Train model with normal weights init
    TrainerGan.train(device=device, generator=generator, discriminator=discriminator,
                     criterion=criterion, learning_rate=learning_rate,
                     real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                     output_path=model_output_path, batch_size=batch_size, weights_init=weights_init)

    i_score, fid_score = Evaluator.evaluate_model(device=device, generator=generator, noise_size=noise_size,
                                                  model_path=f'{model_output_path}/gan_latest',
                                                  output_path=model_output_path, batch_size=batch_size)

    results = [{'name': 'i_score',
                'value': i_score},
               {'name': 'fid',
                'value': fid_score}]

    metadata = None

    return results, metadata
