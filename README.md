# Adversarial Analysis of the RadioML Dataset

still in development.

Couple of notes: 
* Training takes a few hours for the basic experiment and much longer even when using a GPU. Running the experiment with the adversarial data generation will cause a time-out with Google Colab. I receommend running the code on a machine with a dedicated GPU or in the cloud. 
* The logger class in `arml/performance.py` is used to store all of the performances. There might be a better way to save the results moving forward. 
* This code is still in development and therefore should only be used at your own risk! 

# Viewing the Model Training Performances 

The training and validation performances are available in the `logs/fit` directory. [Tensorboard](https://www.tensorflow.org/tensorboard/get_started) is used to monitor these performances while training each model. Note that a new log will be made for each of the runs, so this folder will have many logs. Refer to the Tensorboard documentation for making sense of these files.  
```
$ tensorboard --logdir logs/fit
```
