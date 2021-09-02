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

from .utils import load_radioml, prediction_stats
from .models import vtcnn2, gvn_00

from sklearn.model_selection import KFold

def basic_radioml(file_path:str, n_runs:int=5, verbose:bool=True): 
    """
    """
    train_params = {'dropout': 0.2, 
                    'val_split': 0.2, 
                    'batch_size': 128, 
                    'nb_epoch': 30, 
                    'verbose': 0, 
                    'file_path': 'convmodrecnets_CNN2_0.5.wts.h5'}
    
    X, Y, snrs, mods, encoder = load_radioml(file_path=file_path, shuffle=True)
    
    results_accs, results_aucs, results_ppls = {}, {}, {}
    
    kf = KFold(n_splits=n_runs)
    
    for train_index, test_index in kf.split(X): 
        Xtr, Ytr, Xte, Yte = X[train_index], Y[train_index], X[test_index], Y[test_index]
        mods_tr, snrs_tr, mods_te, snrs_te = mods[train_index], snrs[train_index], mods[test_index], snrs[test_index]

        model, history = vtcnn2(X=Xtr, Y=Ytr, train_param=train_params)
        
        for snr in np.unique(snrs_te): 
            X_c_snr = Xte[snrs_te == snr]
            Yhat = model.predict(X_c_snr) 
            auc, acc, ppl = prediction_stats(Yte[snrs_te==snr], Yhat)
            if snr in results_accs:
                results_aucs[snr] += auc
                results_accs[snr] += acc
                results_ppls[snr] += ppl
            else:
                results_aucs[snr], results_accs[snr], results_ppls[snr] = auc, acc, ppl
    
    for snr in np.unique(snrs): 
        results_aucs[snr] /= n_runs
        results_accs[snr] /= n_runs
        results_ppls[snr] /= n_runs
    
    print(results_accs)
    print(results_aucs)

        # Yhat = model.predict(Xte)
        # auc, acc, ppl = prediction_stats(Yte, Yhat)
        # print(''.join(['ACC: ', str(acc)]))
        # print(''.join(['AUC: ', str(auc)]))
        # print(''.join(['PPL: ', str(ppl)]))
        # print(' ')
    return None 