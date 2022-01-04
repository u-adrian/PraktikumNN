import Trainer


def main():
    Trainer.train(device="GPU", generator="small_gan", discriminator="small_gan", criterion="BCELoss",
                    learning_rate="0.0001", real_img_fake_label="True", num_epochs="200", noise_size="10")
    #Trainer.train(device="GPU", generator="resNet", discriminator="resNet", criterion="BCELoss",
    #              learning_rate="0.0001", real_img_fake_label="True", num_epochs="5", noise_size="100")


if __name__ == "__main__":
    main()