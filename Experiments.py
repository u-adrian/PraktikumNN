import Evaluator
import Trainer
from scores import inception_score_cifar10


def main():
    '''
    unique_name = Trainer.train(device="GPU", generator="small_gan", discriminator="small_gan", criterion="BCELoss",
                                learning_rate="0.0001", real_img_fake_label="True", num_epochs="5", noise_size="20",
                                snapshot_interval="1", name="smallGan01")
    model_path = f'./models/{unique_name}/gan_latest'
    output_path = f'./data/{unique_name}/results'
    Evaluator.create_images(device="GPU", generator="small_gan", noise_size="20", model_path=model_path,
                            output_path=output_path)

    unique_name = Trainer.train(device="GPU", generator="res_net_depth1", discriminator="res_net_depth1",
                                criterion="BCELoss", learning_rate="0.0001", real_img_fake_label="True", num_epochs="5",
                                noise_size="20", snapshot_interval="1", name="resnet01")
    model_path = f'./models/{unique_name}/gan_latest'
    output_path = f'./data/{unique_name}/results'
    Evaluator.create_images(device="GPU", generator="res_net_depth1", noise_size="20", model_path=model_path, output_path=output_path)

    unique_name = Trainer.train(device="GPU", generator="res_net_depth2", discriminator="res_net_depth2",
                                criterion="BCELoss", learning_rate="0.0001", real_img_fake_label="True", num_epochs="5",
                                noise_size="20", snapshot_interval="1", name="resnet02")
    model_path = f'./models/{unique_name}/gan_latest'
    output_path = f'./data/{unique_name}/results'
    Evaluator.create_images(device="GPU", generator="res_net_depth2", noise_size="20", model_path=model_path,
                            output_path=output_path)
    '''
    model_path = 'C:/Dev/test/PraktikumNN/models/resnet02_2022-01-08_18-10-34/gan_latest'
    result_path = f'./models/test/results2/'
    output_path = '###'
    Evaluator.evaluate_model(device="GPU", generator="res_net_depth2", noise_size="20", model_path=model_path,
                            output_path=output_path, result_path=result_path)

if __name__ == "__main__":
    main()
    #inception_score_cifar10(device='GPU', batch_size=100)
