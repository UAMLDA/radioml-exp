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

def load_radioml(file_path): 
    """
    Parameters
    ----------
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1') 

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1,0])
    X, lbl = [], [] 
    for mod in mods:
        for snr in snrs:
            X.append(data[(mod,snr)])
            for i in range(data[(mod,snr)].shape[0]):  lbl.append((mod,snr))

    idx = [i for i in range(len(X))]
    Y = to_onehot(map(lambda x: mods.index(lbl[x][0]), idx))
    X = np.vstack(X)

    return X, Y, snrs, mods

def to_onehot(yy):
    """convert radioml to one-hot vectors 
    """
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1
