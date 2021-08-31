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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_radioml(file_path): 
    """
    Parameters
    ----------
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
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    return X, Y, SNRS, MODS

