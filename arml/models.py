#!/usr/bin/env python 

# Copyright 2021 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import os 
import numpy as np 
import datetime 
import tensorflow as tf 
from tensorflow.keras import models 
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import ZeroPadding2D, Conv2D
# from tensorflow.contrib.tpu.python.tpu import keras_support

tf.compat.v1.disable_eager_execution()

def nn_model(X:np.ndarray, Y:np.ndarray, train_param:dict): 
    """generate a neural network 

    Parameters
    ----------
    X : np.ndarray 
        Training dataset (N, H, W, C)
    Y : np.ndarray 
        Training labels 
    train_param : dict
    """
    if train_param['type'] == 'vtcnn2': 
        model = vtcnn2(X=X, Y=Y, train_param=train_param)
    else: 
        raise(NotImplementedError(''.join([train_param['type'], 'is not implemented.'])))
    return model

def vtcnn2(X:np.ndarray, Y:np.ndarray, train_param:dict): 
    """implementation of the vtcnn2
    
    Parameters
    ----------
    X : np.ndarray 
        Training dataset (N, H, W, C)
    Y : np.ndarray 
        Training labels 
    train_param : dict 
    """
    _, H, W, C = train_param['NHWC']
    
    model = models.Sequential(name='CNN_Architecture')

    model.add(ZeroPadding2D((0,2),
              data_format='channels_last'))

    model.add(Conv2D(256,(1,3),
              activation= 'relu',
              data_format='channels_last',
              input_shape= (H,W,C),
              name = 'ConvLayer1'))
    model.add(Dropout(0.5))
    model.add(Conv2D(80,(2,3),
              activation= 'relu',
              data_format='channels_last', 
              name='ConvLayer2'))
    model.add(Conv2D(256,(1,3),
              activation= 'relu',
              data_format='channels_last',
              name='ConvLayer3'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512,
              activation='relu', 
              name='DenseLayer1'))
    model.add(Dropout(0.5))
    model.add(Dense(256,
              activation='relu', 
              name='DenseLayer2'))
    model.add(Dropout(0.5))
    model.add(Dense(128,
              activation='relu', 
              name='DenseLayer3'))
    model.add(Dense(Y.shape[1],
              activation='softmax', 
              name='Output'))
    
    # split the data into training and validation sets 
    Nval = int(train_param['val_split']*len(Y))
    Xtr, Xval, Ytr, Yval = X[:Nval], X[Nval:], Y[:Nval], Y[Nval:]

    # setup the callbacks: tensorboard, checkpoints and early stopping
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(train_param['file_path'], 
    #                                                          monitor='val_loss', 
    #                                                          verbose=0, 
    #                                                          save_best_only=True, 
    #                                                          mode='auto'),
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(train_param['file_path'], 
                                                            monitor='val_accuracy', 
                                                            verbose=0, 
                                                            save_best_only=True, 
                                                            mode='auto'),
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                          patience=5, 
                                                          verbose=0, 
                                                          mode='auto')

    # compile and build the moedl 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape = (None,H,W,C))

    
    # train the model 
    model.fit(Xtr, Ytr, 
              batch_size=train_param['batch_size'],
              epochs=train_param['nb_epoch'],
              verbose=train_param['verbose'],
              validation_data=(Xval, Yval), 
              callbacks=[
                            tensorboard_callback, 
                            checkpoint_callback, 
                            earlystop_callback
                        ]
              )

    return model 

def get_birnn(X:np.ndarray, Y:np.ndarray): 
    """
    Parameters
    ----------
    """
    return None 

def get_rnn(X:np.ndarray, Y:np.ndarray): 
    """
    Parameters
    ----------
    """
    return None 

def get_cnn(X:np.ndarray, Y:np.ndarray): 
    """
    Parameters
    ----------
    """
    return None 

