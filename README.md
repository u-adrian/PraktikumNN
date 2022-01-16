# PraktikumNN

## Installation and Usage
To get this project running on your computer you first need to install conda.\
[Here](https://docs.anaconda.com/anaconda/install/index.html) you can find the installation guide for Anaconda. 
If you prefer Miniconda you can find an installation guide [here](https://docs.conda.io/en/latest/miniconda.html).

After installing conda you need to create a new environment from the environment.yaml file, 
which you can find in the main directory of this project.

The easiest way to do this is to open a command line in the main directory of the project and execute following command:\
`conda env create -f environment.yaml`\
This will create a new conda environment with the name "praktikum". To work on the new environment 
you need to activate it:\
`conda activate praktikum`\
Now you are able to run the code. To do so, you can execute one of the experiments defined in the "Experiments.py" file.\
An example:\
`python Experiments.py experiment_leaky_vs_normal`

Important information: the experiment "sig_opt_experiment(**kwargs)" cannot be executed in the
same manner used for the other experiments. It won't run on your local machine!


## Arguments for Trainer and Evaluator functions
In this paragraph we explain the arguments you can use for the Trainer and Evaluator functions.\
Some of them are self-explanatory, but some are not. And it is handy to have a list of all input arguments, therefore this paragraph.

- __batch_size__: Defines the size of a training batch.

- __noise_size__: Defines the size of the noise vector used for the generator.

- __num_epochs__: Defines the number of epochs the net is trained.

- __learning_rate__: Defines the learning rate. (We often used "0.0001")

- __criterion__: Defines the loss function. (Currently only "BCELoss" is implemented)

- __real_img_fake_label__: Defines if the training function will additionally train the discriminator with real images but fake labels.
  - Possible Values:
    - False (Default)
    - True

- __pseudo_augment__: Specifies if the algorithm uses data augmentation. 
       Right now only the random horizontal flip is used to augment the dataset.
  - Possible Values:
    - False (Default)
    - True

- __device__: Defines the device on which the model will be trained. (Default: "CPU")
  - Possible Values:
    - "CPU" (Default)
    - "GPU"

- __discriminator__: Defines the net used as discriminator for the model.\
  There is no default value and if you don't give a valid input, the code will throw an error.
  - Possible Values:
    - "small_gan"
    - "res_net_depth1"
    - "res_net_depth1_leaky"
    - "res_net_depth2"

- __generator__: Defines the used as generator for the model.\
  There is no default value and if you don't give a valid input, the code will throw an error.
  - Possible Values:
    - "small_gan"
    - "res_net_depth1"
    - "res_net_depth2" 

- __weights_init__: Defines the method used for the weight initialization of the model.
  - Possible Values:
    - "normal" (Default)
    - "xavier"

- __model_path__: Defines the path to an existing model file. Or the path to the directory of multiple models. This depends on the function.

- __output_path__: Defines the path to a directory where the output (models/results) will be stored.

- __snapshot_interval__: Defines the number of epochs between saving a snapshot of the currently training model.

## Short explanation of the Nets we use

## Description of the Experiments
### Experiment 1: net_configuration
This experiments trains and evaluates different GAN architectures.
We test two core ideas:
First a simple and small architecture using only a few layers called SmallGan.
Second a more complicated structure based on residual connections called ResGan.
Here we use four layers and test one version with one block per layer and one version with two blocks per layer.

### Experiment 2: specialized_training:
This experiment trains and evaluates a GAN with and without special training on real images and false labels.
We test an addition to the training process of the discriminator.
Besides the normal training on real images with their labels and fake images with their labels, the addition includes training on real images with wrong labels.

### Experiment 3: experiment_leaky_vs_normal
This experiment trains and evaluates a GAN with leaky RelU and with normal ReLU as activation in the Discriminator.
We test the impact of the activation function used in the discriminator.
Take note that we only test this for the ResGan architecture.

### Experiment 4: experiment_xavier_vs_normal
This experiment trains and evaluates a GAN with xavier and with normal weight initialization.
We test the impact of the weight initialization for the generator and the discriminator.

### Experiment 5: data_augmentation
This experiment trains and evaluates a GAN with and without augmentation of the training data.
We test the impact of data augmentation for the training results.

### Experiment X: sig_opt_experiment
This Experiment is an attempt to find the best GAN, that our code could produce in a 
reasonable timespan. Meant to be run on the BwUniCluster.



