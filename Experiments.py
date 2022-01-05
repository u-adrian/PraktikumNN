import Evaluator
import Trainer


def main():
    unique_name = Trainer.train_and_get_uniquename(device="GPU", generator="small_gan", discriminator="small_gan",
                                                   criterion="BCELoss", learning_rate="0.0001",
                                                   real_img_fake_label="True", num_epochs="5", noise_size="20",
                                                   snapshot_interval="1", name="smallGan01")

    model_path = f'./models/{unique_name}/gan_latest'
    output_path = f'./data/{unique_name}/results'
    Evaluator.create_images(device="GPU", generator="small_gan", noise_size="20", model_path=model_path, output_path=output_path)
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
if __name__ == "__main__":
    main()