# Adversarial Analysis of the RadioML Dataset


Couple of notes: 
* Training takes a few hours for the basic experiment and much longer even when using a GPU. Running the experiment with the adversarial data generation will cause a time-out with Google Colab. I receommend running the code on a machine with a dedicated GPU or in the cloud. 
* The logger class in `arml/performance.py` is used to store all of the performances. There might be a better way to save the results moving forward. 
* This code is still in development and therefore should only be used at your own risk! 

# Viewing the Model Training Performances 

The training and validation performances are available in the `logs/fit` directory. [Tensorboard](https://www.tensorflow.org/tensorboard/get_started) is used to monitor these performances while training each model. Note that a new log will be made for each of the runs, so this folder will have many logs. Refer to the Tensorboard documentation for making sense of these files.  
```
$ tensorboard --logdir logs/fit
```

# Generating Results 

The Adversarial Robustness Toolbox needs to be installed prior to running the code. Run `pip install -r requirements.txt` to install the dependencies. Once installed, the shell commands below will produce the results. Run each command one at a time if you're using Google Colab. After a command is run then you should restart the Colab session to avoid a timeout. 

```
$ python test_fsgm.py 
$ python test_single_attack.py FastGradientMethod 
$ python test_single_attack.py DeepFool 
$ python test_single_attack.py ProjectedGradientDescent 
$ python test_multiple_attacks.py     # do not run on Google Colab
```