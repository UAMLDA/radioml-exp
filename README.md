# Adversarial Analysis of the RadioML Dataset

still in development. 

# Viewing the Model Training Performances 

The training and validation performances are available in the `logs/fit` directory. [Tensorboard](https://www.tensorflow.org/tensorboard/get_started) is used to monitor these performances while training each model. Note that a new log will be made for each of the runs, so this folder will have many logs. Refer to the Tensorboard documentation for making sense of these files.  
```
$ tensorboard --logdir logs/fit
```
