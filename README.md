# PraktikumNN

##Installation and Usage
To get this project running on your computer you first need to install conda.\
[Here](https://docs.anaconda.com/anaconda/install/index.html) you can find the installation guide for Anaconda. 
If you prefer Miniconda you can find an installation guide [here](https://docs.conda.io/en/latest/miniconda.html).

After installing conda you need to create a new environment from the environment.yaml file, 
which you can find in the main directory of this project.

The easiest way to do this is to open a command line in the main directory of the project and execute following command:\
`conda env create -f environment.yml`\
This will create a new conda environment with the name "praktikum". To work on the new environment 
you need to activate it:\
`conda activate praktikum`\
Now you are able to run the code. To do so, you can execute one of the experiments defined in the "Experiments.py" file.\
An example:\
`python Experiments.py experiment_leaky_vs_normal`


##Arguments for Trainer and Evaluator functions
In this paragraph we explain the arguments you can use for the Trainer and Evaluator functions.\
Some of them are self-explanatory, but some are not. And it is handy to have a list of all input arguments, therefore this paragraph.

- __batch_size__: This argument defines the batch size.
\
<br />
- __criterion__: This defines the loss function, currently only "BCELoss" is implemented.
\
<br />
- __device__: Using this parameter you can define the device on which your model will be trained. Default: "CPU"
  - Possible Values:
    - "CPU" (Default)
    - "GPU"
\
<br />
- __discriminator__: Is used to define the discriminator net for the training.\
  There is no default value and if you don't give a valid input, the code will throw an error.
  - Possible Values:
    - "small_gan"
    - "res_net_depth1"
    - "res_net_depth1_leaky"
    - "res_net_depth2"
\
<br />
- __generator__: Is used to define the generator net.\
  There is no default value and if you don't give a valid input, the code will throw an error.
  - Possible Values:
    - "small_gan"
    - "res_net_depth1"
    - "res_net_depth2"
\
<br />
- __learning_rate__: To define the learning rate. We often used "0.0001".
\
<br />
- __model_path__: Is the path to the model file. Or the path to the directory of multiple models. This depends on the function.
\
<br />
- __noise_size__: Is used to define the size of the noise vector used at the start of the generator, common values are 20 and 100
\
<br />
- __num_epochs__: Defines the number of epochs the net is trained on the dataset
\
<br />
- __output_path__: The folder where the output of the function will be stored
\
<br />
- __real_img_fake_label__: Defines if the training function will train the discriminator with real images but fake labels
  - Possible Values:
    - "True"
    - "False"
\
<br />
- __snapshot_interval__: In which epoch interval do you want to store snapshots of your model? Example: "5" will store snapshots after epochs 0,5,10 and so on.\
                     Note that the epochs start at 0.  
\
<br />
- __weights_init__: Defines the method how the weight will be initialized.
  - Possible Values:
    - "normal" (Default)
    - "xavier"

##Description of the Experiments

