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
from .utils import prediction_stats

class PerfLogger(): 
    """class the track the performances 

    Attributes 
    ----------
    snrs : np.ndarray
        Array of SNRS
    modes : np.ndarray
        Array of modulation types 
    n_classes : np.ndarray
        Number of modulation types 
    n_snrs : np.ndarray
        Number of SNRS 
    accuracy : np.ndarray
        Arracy of accuracy 
    perplexity : np.ndarray
        Arracy of perplexity 
    aucs : np.ndarray
        Arracy of AUCs 
    name : str 
        Name of the logger
    params : dict 
        Training parameters of the experiment 
    count : int 
        Number of runs performed in the experiment 

    Methods 
    -------
    add_scores(Y, Yhat, snr)
        Adds the performance scores from Y and Yhat for the current SNR.
    scale()
        Scale the performances by the number of runs. The class keeps track 
        of the number of runs 
    """

    def __init__(self, name, snrs, mods, params): 
        """initialize the object 
        """
        self.snrs = np.sort(snrs) 
        self.mods = mods
        self.n_classes = len(mods)
        self.n_snrs = len(snrs)
        self.accuracy = np.zeros((self.n_snrs,))
        self.perplexity = np.zeros((self.n_snrs,))
        self.aucs = np.zeros((self.n_snrs,))
        self.name = name 
        self.params = params
        self.count = 0
    
    def add_scores(self, Y, Yhat, snr): 
        """add the current scores to the logger 
        """
        auc, acc, ppl = prediction_stats(Yhat, Y)
        self.accuracy[self.snrs == snr] += acc
        self.perplexity[self.snrs == snr] += ppl
        self.aucs[self.snrs == snr] += auc
        self.count += 1
    
    def scale(self):
        """scale the scores based on the number of runs performed
        """
        self.accuracy /= self.count
        self.perplxity /= self.count 
        self.aucs /= self.count
