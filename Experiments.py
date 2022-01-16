import Evaluator
import Trainer


def net_configurations(experiment_path=f'./experiments/net_configs',
                       device="GPU",
                       criterion="BCELoss",
                       learning_rate="0.0001",
                       real_img_fake_label=True,
                       num_epochs="51",
                       noise_size="20",
                       snapshot_interval="1",
                       batch_size=100,
                       weight_init="normal",
                       augmentation=False):
    # Parameters for experiment
    options = [("SmallGan", "small_gan", "small_gan"),
               ("ResGanDepth1", "res_net_depth1", "res_net_depth1"),
               ("ResGanDepth2", "res_net_depth2", "res_net_depth2")]
    # Run experiments
    for name, generator, discriminator in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weight_init, augmentation)


def specialized_training(experiment_path=f'./experiments/rifl_training',
                         device="GPU",
                         generator="res_net_depth1",
                         discriminator="res_net_depth1",
                         criterion="BCELoss",
                         learning_rate="0.0001",
                         num_epochs="51",
                         noise_size="20",
                         snapshot_interval="1",
                         batch_size=100,
                         weight_init="normal",
                         augmentation=False):
    # Parameters for experiment
    options = [("WithRifl", True),
               ("WithoutRifl", False)]
    # Run experiments
    for name, real_img_fake_label in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weight_init, augmentation)


def leaky_vs_normal_residual_discriminator(experiment_path=f'./experiments/leaky_vs_normal',
                                           device="GPU",
                                           generator="res_net_depth1",
                                           criterion="BCELoss",
                                           learning_rate="0.0001",
                                           real_img_fake_label=True,
                                           num_epochs="51",
                                           noise_size="20",
                                           snapshot_interval="1",
                                           batch_size=100,
                                           weight_init="normal",
                                           augmentation=False):
    # Parameters for experiment
    options = [("LeakyResDiscriminator", "res_net_depth1_leaky"),
               ("ReluResDiscriminator", "res_net_depth1")]
    # Run experiments
    for name, discriminator in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weight_init, augmentation)


def xavier_vs_normal_init(experiment_path=f'./experiments/xavier_vs_normal',
                          device="GPU",
                          generator="res_net_depth1",
                          discriminator="res_net_depth1",
                          criterion="BCELoss",
                          learning_rate="0.0001",
                          real_img_fake_label=True,
                          num_epochs="51",
                          noise_size="20",
                          snapshot_interval="5",
                          batch_size=100,
                          augmentation=False):
    # Parameters for experiment
    options = [("XavierInit", "xavier"),
               ("NormalInit", "normal")]
    # Run experiments
    for name, weight_init in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weight_init, augmentation)


def data_augmentation(experiment_path=f'./experiments/data_aug',
                      device="GPU",
                      generator="small_gan",
                      discriminator="small_gan",
                      criterion="BCELoss",
                      learning_rate="0.0001",
                      real_img_fake_label="True",
                      num_epochs="101",
                      noise_size="20",
                      snapshot_interval="10",
                      batch_size=100,
                      weight_init="normal"):
    # Parameters for experiment
    options = [("WithoutAugmentation", False),
               ("WithAugmentation", True)]
    # Run experiments
    for name, augmentation in options:
        _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                            real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                            weight_init, augmentation)


def _execute_experiment(experiment_path, name, device, generator, discriminator, criterion, learning_rate,
                        real_img_fake_label, num_epochs, noise_size, snapshot_interval, batch_size,
                        weight_init, augmentation):
    # Train model
    model_path = f'./{experiment_path}/models/{name}'
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=model_path,
                  batch_size=batch_size, weight_init=weight_init, pseudo_augmentation=augmentation)
    # Evaluate model
    scores_path = f'./{experiment_path}/models/{name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)


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

    weight_init = model_args['weight_init']

    # Train model with normal weight init
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  output_path=model_output_path, batch_size=batch_size, weight_init=weight_init)

    i_score, fid_score = Evaluator.evaluate_model(device=device, generator=generator, noise_size=noise_size,
                                                  model_path=f'{model_output_path}/gan_latest',
                                                  output_path=model_output_path, batch_size=batch_size)

    results = [{'name': 'i_score',
                'value': i_score},
               {'name': 'fid',
                'value': fid_score}]

    metadata = None

    return results, metadata
