# PraktikumNN

Welcome to our project in which we try to generate images according to the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data set by using different Conditional Generative Adversarial Nets. If you don't read this on GitHub we encourage you to have a look
at our [GitHub Repository](https://github.com/u-adrian/PraktikumNN).


## Explanation of the Nets we use
In this project we use 2 types of nets which can be found in the "Nets" directory. The first one is a simple implementation, while the second one includes residual connections.
The basic idea follows [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf).
### SmallGan
This architecture is influenced by the [Deep Convolutional GAN (DCGAN)](https://arxiv.org/abs/1511.06434).

#### Small Generator
We extended this version to incorporate a condition, representing the desired class label of the generated image.
The generator therefore receives a latent vector consisting of noise and condition. It uses a stack of transposed convolutions with batch norm and ReLU activations to create a 32 by 32 colored image.

#### The structure looks as follows:
<img src="images/SmallGenerator.jpeg">

#### Small Discriminator
The discriminator consists of a feature extractor and a linear classifier.
The feature extractor receives a 32 by 32 colored image and uses a stack of normal convolutions with batch norm and ReLU activations. The resulting features are given to the classifier together with the respective condition, which then outputs a value between 0 and 1.

#### The structure looks as follows:
<img src="images/SmallDiscriminator.jpeg">

### ResGan
This architecture is influenced by the [Deep Residual Net (ResNet)](https://arxiv.org/abs/1512.03385).

#### Residual Discriminator
We implemented a small modified ResNet as a discriminator and added the condition as one-hot-encoded feature maps to the 32 by 32 input image.
It uses residual blocks to form residual layers where each new layer performs downsampling (halve size, double channels).

<img src="images/ResidualLayers.jpeg">

We use three layers with each one or alternatively two blocks per layer.
Together with a gate and a decoder, this constructs the discriminator.
#### Depth1
<img src="images/ResidualDiscriminator.jpeg">

#### Depth 2
<img src="images/ResidualDiscriminator2.jpeg">

#### The final Residual Discriminator looks as follows
<img src="images/ResidualDiscriminatorSummary.jpeg">

#### Residual Generator
For the generator we inverted this idea by defining a new transposed residual net.
It uses transposed convolutions instead of normal convolutions and upsampling (double size, halve channels) between layers instead of downsampling.

<img src="images/TransposedResidualLayers.jpeg">

 Again, we use three layers with each one or alternatively two blocks per layer. Combined with a gate and a decoder, this constructs the generator.
#### Depth 1
<img src="images/ResidualGenerator.jpeg">

#### Depth 2
<img src="images/ResidualGenerator2.jpeg">

#### The final Residual Generator looks as follows
<img src="images/ResidualGeneratorSummary.jpeg">

### Auto Encoder
This architecture is only used for pretraining a generator network. It is not a conditional GAN itself.
Here we use a simple encoder to encode an image into a feature vector. A generator is trained to reconstruct the feature vector and respective condition to the original image.
As already said, this is not a network used to create images and only meant as an experiment to test if pretraining a generator in such a way affects the training of a full GAN.


## Description of the Experiments
### Experiment 1: net_configuration
This experiment trains and evaluates different GAN architectures.
We test two core ideas:
First a simple and small architecture using only a few layers called SmallGan.
Second a more complicated structure based on residual connections called ResGan.
Here we use three layers and test one version with one block per layer and one version with two blocks per layer.

### Experiment 2: specialized_training:
This experiment trains and evaluates a GAN with and without special training on real images and false labels.
We test an addition to the training process of the discriminator.
Besides the normal training on real images with their labels and fake images with their labels, the addition includes training on real images with wrong labels.

### Experiment 3: leaky_vs_normal_residual_discriminator
This experiment trains and evaluates a GAN with leaky ReLU and with normal ReLU as activation in the Discriminator.
We test the impact of the activation function used in the discriminator.
Take note that we only test this for the ResGan architecture.

### Experiment 4: xavier_vs_normal_init
This experiment trains and evaluates a GAN with xavier and with normal weight initialization.
We test the impact of the weight initialization for the generator and the discriminator.

### Experiment 5: data_augmentation
This experiment trains and evaluates a GAN with and without augmentation of the training data.
We test the impact of data augmentation for the training results.

### Experiment 6: generator_pretraining
This experiment trains and evaluates a GAN with and without pretraining of the training generator.
We test the impact of pretraining the generator as an autoencoder for the training results.

### Experiment X: sig_opt_experiment (results still pending)
This Experiment is an attempt to find the best GAN that our code could produce in a 
reasonable timespan. Meant to be run on the [BwUniCluster](https://wiki.bwhpc.de/e/Category:BwUniCluster_2.0). 
To accomplish our goal we use the code from the repository "cluster-hyperopt" developed by "aimat-lab", 
a tool called [SigOpt](https://sigopt.com/) and the BwUniCluster.
Sadly, we can't provide a link to the "aimat-lab/cluster-hyperopt" repository, since this project isn't public yet.
SigOpt is a tool for hyperparameter optimization.

What "cluster-hyperopt" does:
1) getting hyper-parameter suggestions from SigOpt
2) training and evaluating the model with the suggested parameters
3) sending the scores to SigOpt and start at 1) again

In our case this loop will be repeated 30 times.


## Current Results
We used the following configurations for our currently best working conditional GAN:
- generator = "res_net_depth1"
- discriminator = "res_net_depth1"
- criterion = "BCELoss"
- learning_rate = 0.0001
- real_img_fake_label = True
- num_epochs = 31
- noise_size = 20
- batch_size = 100
- weight_init = "normal"
- augmentation = True
- pretrained = False

We achieved following scores with this setup:
- inception score: 2.09
- frechet inception distance: 173.1

Here you can see some generated images:

#### plane

<img src="images/generated/plane_0.png" width="64" height="64">   <img src="images/generated/plane_1.png" width="64" height="64">   <img src="images/generated/plane_2.png" width="64" height="64">   <img src="images/generated/plane_3.png" width="64" height="64">   <img src="images/generated/plane_4.png" width="64" height="64">   <img src="images/generated/plane_5.png" width="64" height="64">   <img src="images/generated/plane_6.png" width="64" height="64">   <img src="images/generated/plane_7.png" width="64" height="64">   <img src="images/generated/plane_8.png" width="64" height="64">   <img src="images/generated/plane_9.png" width="64" height="64">   

#### car

<img src="images/generated/car_0.png" width="64" height="64">   <img src="images/generated/car_1.png" width="64" height="64">   <img src="images/generated/car_2.png" width="64" height="64">   <img src="images/generated/car_3.png" width="64" height="64">   <img src="images/generated/car_4.png" width="64" height="64">   <img src="images/generated/car_5.png" width="64" height="64">   <img src="images/generated/car_6.png" width="64" height="64">   <img src="images/generated/car_7.png" width="64" height="64">   <img src="images/generated/car_8.png" width="64" height="64">   <img src="images/generated/car_9.png" width="64" height="64">   

#### bird

<img src="images/generated/bird_0.png" width="64" height="64">   <img src="images/generated/bird_1.png" width="64" height="64">   <img src="images/generated/bird_2.png" width="64" height="64">   <img src="images/generated/bird_3.png" width="64" height="64">   <img src="images/generated/bird_4.png" width="64" height="64">   <img src="images/generated/bird_5.png" width="64" height="64">   <img src="images/generated/bird_6.png" width="64" height="64">   <img src="images/generated/bird_7.png" width="64" height="64">   <img src="images/generated/bird_8.png" width="64" height="64">   <img src="images/generated/bird_9.png" width="64" height="64">   

#### cat

<img src="images/generated/cat_0.png" width="64" height="64">   <img src="images/generated/cat_1.png" width="64" height="64">   <img src="images/generated/cat_2.png" width="64" height="64">   <img src="images/generated/cat_3.png" width="64" height="64">   <img src="images/generated/cat_4.png" width="64" height="64">   <img src="images/generated/cat_5.png" width="64" height="64">   <img src="images/generated/cat_6.png" width="64" height="64">   <img src="images/generated/cat_7.png" width="64" height="64">   <img src="images/generated/cat_8.png" width="64" height="64">   <img src="images/generated/cat_9.png" width="64" height="64">   

#### deer

<img src="images/generated/deer_0.png" width="64" height="64">   <img src="images/generated/deer_1.png" width="64" height="64">   <img src="images/generated/deer_2.png" width="64" height="64">   <img src="images/generated/deer_3.png" width="64" height="64">   <img src="images/generated/deer_4.png" width="64" height="64">   <img src="images/generated/deer_5.png" width="64" height="64">   <img src="images/generated/deer_6.png" width="64" height="64">   <img src="images/generated/deer_7.png" width="64" height="64">   <img src="images/generated/deer_8.png" width="64" height="64">   <img src="images/generated/deer_9.png" width="64" height="64">   

#### dog

<img src="images/generated/dog_0.png" width="64" height="64">   <img src="images/generated/dog_1.png" width="64" height="64">   <img src="images/generated/dog_2.png" width="64" height="64">   <img src="images/generated/dog_3.png" width="64" height="64">   <img src="images/generated/dog_4.png" width="64" height="64">   <img src="images/generated/dog_5.png" width="64" height="64">   <img src="images/generated/dog_6.png" width="64" height="64">   <img src="images/generated/dog_7.png" width="64" height="64">   <img src="images/generated/dog_8.png" width="64" height="64">   <img src="images/generated/dog_9.png" width="64" height="64">   

#### frog

<img src="images/generated/frog_0.png" width="64" height="64">   <img src="images/generated/frog_1.png" width="64" height="64">   <img src="images/generated/frog_2.png" width="64" height="64">   <img src="images/generated/frog_3.png" width="64" height="64">   <img src="images/generated/frog_4.png" width="64" height="64">   <img src="images/generated/frog_5.png" width="64" height="64">   <img src="images/generated/frog_6.png" width="64" height="64">   <img src="images/generated/frog_7.png" width="64" height="64">   <img src="images/generated/frog_8.png" width="64" height="64">   <img src="images/generated/frog_9.png" width="64" height="64">   

#### horse

<img src="images/generated/horse_0.png" width="64" height="64">   <img src="images/generated/horse_1.png" width="64" height="64">   <img src="images/generated/horse_2.png" width="64" height="64">   <img src="images/generated/horse_3.png" width="64" height="64">   <img src="images/generated/horse_4.png" width="64" height="64">   <img src="images/generated/horse_5.png" width="64" height="64">   <img src="images/generated/horse_6.png" width="64" height="64">   <img src="images/generated/horse_7.png" width="64" height="64">   <img src="images/generated/horse_8.png" width="64" height="64">   <img src="images/generated/horse_9.png" width="64" height="64">   

#### ship

<img src="images/generated/ship_0.png" width="64" height="64">   <img src="images/generated/ship_1.png" width="64" height="64">   <img src="images/generated/ship_2.png" width="64" height="64">   <img src="images/generated/ship_3.png" width="64" height="64">   <img src="images/generated/ship_4.png" width="64" height="64">   <img src="images/generated/ship_5.png" width="64" height="64">   <img src="images/generated/ship_6.png" width="64" height="64">   <img src="images/generated/ship_7.png" width="64" height="64">   <img src="images/generated/ship_8.png" width="64" height="64">   <img src="images/generated/ship_9.png" width="64" height="64">   

#### truck

<img src="images/generated/truck_0.png" width="64" height="64">   <img src="images/generated/truck_1.png" width="64" height="64">   <img src="images/generated/truck_2.png" width="64" height="64">   <img src="images/generated/truck_3.png" width="64" height="64">   <img src="images/generated/truck_4.png" width="64" height="64">   <img src="images/generated/truck_5.png" width="64" height="64">   <img src="images/generated/truck_6.png" width="64" height="64">   <img src="images/generated/truck_7.png" width="64" height="64">   <img src="images/generated/truck_8.png" width="64" height="64">   <img src="images/generated/truck_9.png" width="64" height="64">


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
`python -c 'import Experiments; Experiments.net_configurations()'`

Important information: the experiment "sig_opt_experiment(**kwargs)" cannot be executed in the
same manner used for the other experiments. It won't run on your local machine!


## Arguments for Trainer and Evaluator Functions
In this paragraph we explain the arguments you can use for the Trainer and Evaluator functions.\
Some of them are self-explanatory, but some are not. And it is handy to have a list of all input arguments, therefore this paragraph.

- __batch_size__: Defines the size of a training batch.

- __noise_size__: Defines the size of the noise vector used for the generator.

- __num_epochs__: Defines the number of epochs the net is trained.

- __learning_rate__: Defines the learning rate.

- __criterion__: Defines the loss function. (Currently only "BCELoss" is implemented)

- __real_img_fake_label__: Defines whether the training function will additionally train the discriminator with real images but fake labels.
  - Possible Values:
    - False (Default)
    - True

- __pretraining__: Defines whether training function will use a pretrained generator or not. Path to pretrained model needs to be specified in __model_path__. 
  - Possible Values:
    - False (Default)
    - True

- __augmentation__: Specifies if the algorithm uses data augmentation. 
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

- __experiments_path__: Defines the path to a directory where the experiment results will be stored.

- __output_path__: Defines the path to a directory where the output (models/results) will be stored.

- __snapshot_interval__: Defines the number of epochs between saving a snapshot of the currently training model.
