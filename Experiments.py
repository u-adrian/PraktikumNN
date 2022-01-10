import Evaluator
import Trainer
from scores import inception_score_cifar10


def main():

    # unique_name = Trainer.train(device="GPU", generator="res_net_depth1", discriminator="res_net_depth1", criterion="BCELoss",
    #                             learning_rate="0.00001", real_img_fake_label="False", num_epochs="100", noise_size="20",
    #                             snapshot_interval="5", name="resNet01", batch_size=100)
    # scores_path = f'C:/Dev/test/PraktikumNN/models/resNet01_2022-01-09_18-37-32/snapshots'
    #
    # Evaluator.evaluate_multiple_models(device="GPU", generator="res_net_depth1", noise_size="20", model_path=scores_path,
    #                                    output_path=scores_path, batch_size=100)

    unique_name = Trainer.train(device="GPU", generator="res_net_depth2", discriminator="res_net_depth2",
                                criterion="BCELoss",
                                learning_rate="0.00001", real_img_fake_label="False", num_epochs="100", noise_size="20",
                                snapshot_interval="5", name="resNet02", batch_size=100)
    scores_path = f'C:/Dev/test/PraktikumNN/models/{unique_name}/snapshots'

    Evaluator.evaluate_multiple_models(device="GPU", generator="res_net_depth2", noise_size="20",
                                       model_path=scores_path,
                                       output_path=scores_path, batch_size=100)

    unique_name = Trainer.train(device="GPU", generator="res_net_depth1", discriminator="res_net_depth1",
                                criterion="BCELoss",
                                learning_rate="0.00001", real_img_fake_label="True", num_epochs="100", noise_size="20",
                                snapshot_interval="5", name="resNet01rifl", batch_size=100)
    scores_path = f'C:/Dev/test/PraktikumNN/models/{unique_name}/snapshots'

    Evaluator.evaluate_multiple_models(device="GPU", generator="res_net_depth1", noise_size="20",
                                       model_path=scores_path,
                                       output_path=scores_path, batch_size=100)

    unique_name = Trainer.train(device="GPU", generator="res_net_depth2", discriminator="res_net_depth2",
                                criterion="BCELoss",
                                learning_rate="0.00001", real_img_fake_label="True", num_epochs="100", noise_size="20",
                                snapshot_interval="5", name="resNet02rifl", batch_size=100)
    scores_path = f'C:/Dev/test/PraktikumNN/models/{unique_name}/snapshots'

    Evaluator.evaluate_multiple_models(device="GPU", generator="res_net_depth2", noise_size="20",
                                       model_path=scores_path,
                                       output_path=scores_path, batch_size=100)
if __name__ == "__main__":
    main()
