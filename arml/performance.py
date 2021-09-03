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

    def __init__(self, name:str, snrs:np.ndarray, mods:np.ndarray, params:dict): 
        """initialize the object 

        Parameters
        ----------
        name : str 
            Name of the logger 
        snrs : np.ndarray 
            Array of SNRs 
        mods : np.ndarray 
            Array of MODs 
        params : dict 
            Dictionary of training parameters 
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
        self.number_correct = 0. 
        self.number_instances_processed = 0.
        self.overall_accuracy = 0
    
    def add_scores(self, Y:np.ndarray, Yhat:np.ndarray, snr:float): 
        """add the current scores to the logger 

        Parameters
        ----------
        Y : np.ndarray 
            Ground truth labels 
        Yhat : np.ndarray 
            Predictions 
        snr : int 
            SNR level from the predictions 
        """
        auc, acc, ppl = prediction_stats(Y, Yhat)
        self.accuracy[self.snrs == snr] += acc
        self.perplexity[self.snrs == snr] += ppl
        self.aucs[self.snrs == snr] += auc
        self.count += 1
        self.number_instances_processed += len(Y)
        self.number_correct += len(Y)*acc
    
    def finalize(self):
        """scale the scores based on the number of runs performed
        """
        self.accuracy /= self.count
        self.perplxity /= self.count 
        self.aucs /= self.count
        self.overall_accuracy = self.number_correct/self.number_instances_processed

class AdversarialPerfLogger(): 
    """
    """
    
    def __init__(self, name:str, snrs:np.ndarray, mods:np.ndarray, params:dict): 
        """
        """
        self.snrs = np.sort(snrs) 
        self.mods = mods
        self.n_classes = len(mods)
        self.n_snrs = len(snrs)
        
        self.accuracy = np.zeros((self.n_snrs,))
        self.perplexity = np.zeros((self.n_snrs,))
        self.aucs = np.zeros((self.n_snrs,))
        
        self.accuracy_fgsm = np.zeros((self.n_snrs,))
        self.perplexity_fgsm = np.zeros((self.n_snrs,))
        self.aucs_fgsm = np.zeros((self.n_snrs,))

        self.accuracy_deep = np.zeros((self.n_snrs,))
        self.perplexity_deep = np.zeros((self.n_snrs,))
        self.aucs_deep = np.zeros((self.n_snrs,))

        self.accuracy_pgd = np.zeros((self.n_snrs,))
        self.perplexity_pgd = np.zeros((self.n_snrs,))
        self.aucs_pgd = np.zeros((self.n_snrs,))

        self.name = name 
        self.params = params
        self.count = 0
    
    def add_scores(self, Y, Yhat, Yhat_fgsm, Yhat_deep, Yhat_pgd, snr): 
        """
        """
        auc, acc, ppl = prediction_stats(Y, Yhat)
        self.accuracy[self.snrs == snr] += acc
        self.perplexity[self.snrs == snr] += ppl
        self.aucs[self.snrs == snr] += auc

        auc, acc, ppl = prediction_stats(Y, Yhat_fgsm)
        self.accuracy_fgsm[self.snrs == snr] += acc
        self.perplexity_fgsm[self.snrs == snr] += ppl
        self.aucs_fgsm[self.snrs == snr] += auc

        auc, acc, ppl = prediction_stats(Y, Yhat_deep)
        self.accuracy_deep[self.snrs == snr] += acc
        self.perplexity_deep[self.snrs == snr] += ppl
        self.aucs_deep[self.snrs == snr] += auc

        auc, acc, ppl = prediction_stats(Y, Yhat_pgd)
        self.accuracy_pgd[self.snrs == snr] += acc
        self.perplexity_pgd[self.snrs == snr] += ppl
        self.aucs_pgd[self.snrs == snr] += auc

        self.count += 1

    def finalize(self): 
        """
        """
        self.accuracy /= self.count
        self.perplxity /= self.count 
        self.aucs /= self.count

        self.accuracy_fgsm /= self.count
        self.perplxity_fgsm /= self.count 
        self.aucs_fgsm /= self.count

        self.accuracy_deep /= self.count
        self.perplxity_deep /= self.count 
        self.aucs_deep /= self.count

        self.accuracy_pgd /= self.count
        self.perplxity_pgd /= self.count 
        self.aucs_pgd /= self.count
