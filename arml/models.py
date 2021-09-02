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
from tensorflow.keras.layers import Convolution2D, MaxPool2D, ZeroPadding2D, Conv1D, MaxPooling1D, Conv2D

tf.compat.v1.disable_eager_execution()


def gvn_00(X:np.ndarray, Y:np.ndarray, train_param:dict):
    """does not work yet
    """
    n_class = Y.shape[1]
    in_shp = list(X.shape[1:])

    model = models.Sequential()
    model.add(Conv1D(filters=128,
                 kernel_size=16,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 data_format='channels_first',
                 name="conv1", 
                 kernel_initializer='glorot_uniform',
                 input_shape=in_shp))
    model.add(Dropout(rate=.5))
    model.add(MaxPooling1D(pool_size=1,padding='valid', name="pool1"))
    model.add(Conv1D(filters=128,
                 kernel_size=8,
                 strides=4,
                 padding='valid', 
                 activation="relu", 
                 data_format='channels_first',
                 name="conv2", 
                 kernel_initializer='glorot_uniform'))
    model.add(Dropout(rate=.5))
    model.add(MaxPooling1D(pool_size=1, padding='valid', name="pool2"))
    model.add(Conv1D(filters=128,
                 kernel_size=2,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 name="conv3", 
                 data_format='channels_first',
                 kernel_initializer='glorot_uniform'))
    model.add(Dropout(rate=.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(rate=.5))
    model.add(Dense(n_class, kernel_initializer='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    Nval = np.int16(train_param['val_split']*len(Y))
    Xtr, Xval, Ytr, Yval = X[:Nval], X[Nval:], Y[:Nval], Y[Nval:]

    history = model.fit(Xtr, Ytr,
        batch_size=train_param['batch_size'],
        epochs=train_param['nb_epoch'],
        verbose=train_param['verbose'],
        validation_data=(Xval, Yval), 
        callbacks = [
        tf.keras.callbacks.ModelCheckpoint(train_param['file_path'], monitor='val_loss', \
            verbose=0, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
    model.load_weights(train_param['file_path'])

    return model, history 


    return model


def vtcnn2(X:np.ndarray, Y:np.ndarray, train_param:dict): 
    """implementation of the vtcnn2
    Parameters
    ----------
    """
    N, H, W, C = train_param['NHWC']
    
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
    model.add(Dense(11,
              activation='softmax', 
              name='Output'))

    # model = models.Sequential()
    # model.add(Reshape([1]+in_shp, input_shape=in_shp))

    # model.add(ZeroPadding2D((0, 2)))
    # model.add(Convolution2D(256, 1, 3, activation="relu", name="conv1"))
    # model.add(Dropout(train_param['dropout']))

    # model.add(ZeroPadding2D((0, 2)))
    # model.add(Convolution2D(128, 1, 3, activation="relu", name="conv2"))
    # model.add(Dropout(train_param['dropout']))

    # model.add(Flatten())
    # model.add(Dense(256, activation='relu', name="dense1"))
    # model.add(Dropout(train_param['dropout']))
    
    # model.add(Dense(128, activation='relu', name="dense3"))
    # model.add(Dropout(train_param['dropout']))
    
    # model.add(Dense(Y.shape[1], name="dense2" ))
    # model.add(Activation('softmax'))
    # model.add(Reshape([Y.shape[1]]))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.build(input_shape = (None,H,W,C))


    Nval = int(train_param['val_split']*len(Y))
    Xtr, Xval, Ytr, Yval = X[:Nval], X[Nval:], Y[:Nval], Y[Nval:]

    history = model.fit(Xtr, Ytr,
        batch_size=train_param['batch_size'],
        epochs=train_param['nb_epoch'],
        verbose=train_param['verbose'],
        validation_data=(Xval, Yval))#, 
        #use_multiprocessing=True,
        #callbacks = [
        #tf.keras.callbacks.ModelCheckpoint(train_param['file_path'], monitor='val_accuracy', \
        #    verbose=0, save_best_only=True, mode='auto'),
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    #])
    # model.load_weights(train_param['file_path'])

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

