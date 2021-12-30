import MainClass


def main():
    MainClass.train(device="GPU", generator="small_gan", discriminator="small_gan", criterion="BCELoss", learning_rate="0.0001")


if __name__ == "__main__":
    main()