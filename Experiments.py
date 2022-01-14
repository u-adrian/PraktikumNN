import Evaluator
import Trainer


def experiment_leaky_vs_normal():
    # Parameters
    device = "GPU"
    generator = "res_net_depth1"
    criterion = "BCELoss"
    learning_rate = "0.0001"
    real_img_fake_label = "False"
    num_epochs = "51"
    noise_size = "20"
    snapshot_interval = "1"
    batch_size = 100
    # Important
    name = "resNet01Leaky"
    discriminator = "res_net_depth1_leaky"

    # Train model with Leaky ReLu
    model_path = f'./models/{name}'
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=model_path, batch_size=batch_size)
    scores_path = f'./models/{name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)

    # Changes
    name = "resNet01NonLeaky"
    discriminator = "res_net_depth1"
    model_path = f'./models/{name}'

    # Train model with normal ReLu
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=model_path, batch_size=batch_size)
    scores_path = f'./models/{name}/snapshots'
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

    # Train model with xavier weight init
    model_path = f'./models/{name}'
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=model_path, batch_size=batch_size,
                  weight_init=weight_init)
    scores_path = f'./models/{name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)

    # Changes
    name = "resNet01NormalWeightInit"
    weight_init = "normal"
    model_path = f'./models/{name}'

    # Train model with normal weight init
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=model_path, batch_size=batch_size,
                  weight_init=weight_init)
    scores_path = f'./models/{name}/snapshots'
    Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                       model_path=scores_path, output_path=scores_path, batch_size=batch_size)


def inception_stability():
    # Parameters
    device = "GPU"
    generator = "res_net_depth1"
    discriminator = "res_net_depth1"
    criterion = "BCELoss"
    learning_rate = "0.0001"
    real_img_fake_label = "False"
    num_epochs = "10"
    noise_size = "20"
    snapshot_interval = "1"
    batch_size = 100
    # Important
    name = "resNet01"

    # Train model with xavier weight init
    model_path = f'./inception_stability/model/{name}'
    Trainer.train(device=device, generator=generator, discriminator=discriminator,
                  criterion=criterion, learning_rate=learning_rate,
                  real_img_fake_label=real_img_fake_label, num_epochs=num_epochs, noise_size=noise_size,
                  snapshot_interval=snapshot_interval, output_path=model_path, batch_size=batch_size)

    output_path = './inception_stability/data/scores'
    snapshots_path = f'./inception_stability/model/{name}/snapshots'
    for i in range(10):
        scores_path = f'{output_path}/scores_iteration{i}'
        Evaluator.evaluate_multiple_models(device=device, generator=generator, noise_size=noise_size,
                                           model_path=snapshots_path, output_path=scores_path, batch_size=batch_size)

def sig_opt_experiment(**kwargs):
    # Parameters
    device = "GPU"                      #constant
    criterion = "BCELoss"               #constant
    batch_size = 100                    #constant
    model_output_path = kwargs['model_output_path']

    model_args = kwargs['suggestion']
    ### 6 params in kwargs ###
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
                                             model_path=f'{model_output_path}/gan_latest', output_path=model_output_path, batch_size=batch_size)

    results = [{'name': 'i_score',
                'value': i_score},
               {'name': 'fid',
                'value': fid_score}]

    metadata = None

    return results, metadata


def main():
    experiment_leaky_vs_normal()
    # experiment_xavier_vs_normal()
    # inception_stability()

if __name__ == "__main__":
    main()
