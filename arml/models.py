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

import numpy as np 

import tensorflow as tf 
from tensorflow.keras import models 
from tensorflow.keras.layers import Reshape, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPool2D, ZeroPadding2D

def get_vtcnn2(X:np.ndarray, Y:np.ndarray, train_param:dict): 
    """implementation of the vtcnn2
    Parameters
    ----------
    """
    in_shp = list(X.shape[1:])

    model = models.Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", \
        name="conv1", init='glorot_uniform'))
    model.add(Dropout(train_param['dropout']))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu", \
        name="conv2", init='glorot_uniform'))
    model.add(Dropout(train_param['dropout']))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
    model.add(Dropout(train_param['dropout']))
    model.add(Dense(Y.shape[1], init='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.add(Reshape([Y.shape[1]]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    Nval = np.int16(train_param['val_split']*len(Y))
    Xtr, Xval, Ytr, Yval = X[:Nval], X[Nval:], Y[:Nval], Y[Nval]

    history = model.fit(Xtr, Ytr,
        batch_size=train_param['batch_size'],
        nb_epoch=train_param['nb_epoch'],
        show_accuracy=False,
        verbose=2,
        validation_data=(Xval, Yval), 
        callbacks = [
        tf.keras.callbacks.ModelCheckpoint(train_param['file_path'], monitor='val_loss', \
            verbose=0, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
    model.load_weights(train_param['file_path'])

    return model, history 

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

