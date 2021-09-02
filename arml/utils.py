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

import pickle 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

def load_radioml(file_path:str, shuffle:bool=True): 
    """load the radioml dataset from a pickle file

    Parameters
    ----------
    file_path : str 
        File path to the radioml pickle file 
    shuffle : bool 
        Permutate the data? [default=True]
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1') 

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1,0])
    X, Y, MODS, SNRS = [], [], [], [] 
    for mod in mods:
        for snr in snrs:
            X.append(data[(mod,snr)])
            for i in range(data[(mod,snr)].shape[0]):  Y.append(mod)
            for i in range(data[(mod,snr)].shape[0]):  MODS.append(mod)
            for i in range(data[(mod,snr)].shape[0]):  SNRS.append(snr)
            
    X = np.vstack(X)
    Y = np.array(Y)

    # convert the MODS to the labels using one hot encoding
    encoder = OneHotEncoder(sparse=False).fit(Y.reshape(-1,1))
    Y = encoder.transform(Y.reshape(-1,1))

    SNRS, MODS = np.array(SNRS), np.array(MODS)

    # shuffle the data?
    if shuffle: 
        i = np.random.permutation(X.shape[0])
        X, Y, MODS, SNRS = X[i], Y[i], MODS[i], SNRS[i]

    return X, Y, SNRS, MODS, encoder


def prediction_stats(Y, Yhat):
    """prediction statistics

    Parameters 
    ----------
    Y : np.ndarray 
        Ground truth labels 
    Yhat : np.ndarray 
        Predicted probabilities

    Returns
    -------
    (auc, acc, ppl) : tuple(float, float, float) 
    """
    auc = roc_auc_score(Y, Yhat)
    acc = (np.argmax(Y, axis=1) == np.argmax(Yhat, axis=1)).sum()/len(Y)
    ppl = 2**(-(Y*Yhat).sum(axis=1).mean())
    return auc, acc, ppl 