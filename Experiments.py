import Evaluator
import Trainer
from scores import inception_score_cifar10


def main():
    '''

    unique_name = Trainer.train_and_get_uniquename(device="GPU", generator="small_gan", discriminator="small_gan",
                                                   criterion="BCELoss", learning_rate="0.0001",
                                                   real_img_fake_label="True", num_epochs="5", noise_size="20",
                                                   snapshot_interval="1", name="smallGan01")

    model_path = f'./models/{unique_name}/gan_latest'
    output_path = f'./data/{unique_name}/results'
    Evaluator.create_images(device="GPU", generator="small_gan", noise_size="20", model_path=model_path, output_path=output_path)
    '''
    '''
    unique_name = Trainer.train_and_get_uniquename(device="GPU", generator="res_net", discriminator="res_net",
                                                   criterion="BCELoss", learning_rate="0.0001",
                                                   real_img_fake_label="True", num_epochs="5", noise_size="20",
                                                   snapshot_interval="1", name="resnet01")
    model_path = f'./models/{unique_name}/gan_latest'
    output_path = f'./data/{unique_name}/results'
    Evaluator.create_images(device="GPU", generator="res_net", noise_size="20", model_path=model_path,
                            output_path=output_path)
    '''

    model_path = f'./models/gan_Resnet_depth2_45E'
    result_path = f'./models/test/results/'
    output_path = f'./models/resnet01_2022-01-06_09-00-52_results/results'
    Evaluator.evaluate_model(device="CPU", generator="res_net", noise_size="20", model_path=model_path,
                            output_path=output_path, result_path=result_path)

if __name__ == "__main__":
    #main()
    inception_score_cifar10(device='CPU')
