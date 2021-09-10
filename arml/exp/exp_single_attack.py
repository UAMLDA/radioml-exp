
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

from ..utils import load_radioml
from ..models import nn_model
from ..performance import AdversarialPerfLogger
from ..adversarial_data import generate_aml_data

from sklearn.model_selection import KFold

def experiment_single_adversarial(file_path:str,
                           n_runs:int=5, 
                           verbose:int=1, 
                           scenario:str='A', 
                           attack_type:str='FastGradientMethod', 
                           train_params:dict={}, 
                           train_adversary_params:dict={}, 
                           logger_name:str='aml_radioml_vtcnn2_vtcnn2_scenario_A',
                           output_path:str='outputs/aml_vtcnn2_vtcnn2_scenario_A_radioml.pkl'): 
    """
    """

    X, Y, snrs, mods, _ = load_radioml(file_path=file_path, shuffle=True)
    C = 1
    N, H, W = X.shape
    X = X.reshape(N, H, W, C)

    if len(train_params) == 0:
        train_params = {'type': 'vtcnn2', 
                        'dropout': 0.5, 
                        'val_split': 0.9, 
                        'batch_size': 1024, 
                        'nb_epoch': 50, 
                        'verbose': verbose, 
                        'NHWC': [N, H, W, C],
                        'tpu': False, 
                        'file_path': 'convmodrecnets_CNN2_0.5.wts.h5'}
    
    if len(train_adversary_params) == 0:
        train_adversary_params = {'type': 'vtcnn2', 
                                  'dropout': 0.5, 
                                  'val_split': 0.9, 
                                  'batch_size': 1024, 
                                  'nb_epoch': 50, 
                                  'verbose': verbose, 
                                  'NHWC': [N, H, W, C],
                                  'epsilon': 0.15, 
                                  'file_path': 'convmodrecnets_adversary_CNN2_0.5.wts.h5'}
    
    # initialize the performances to empty 
    result_logger = AdversarialPerfLogger(name=logger_name, 
                                          snrs=np.unique(snrs), 
                                          mods=np.unique(mods), 
                                          params=[train_params, train_adversary_params])
    
    kf = KFold(n_splits=n_runs)
    
    for train_index, test_index in kf.split(X): 
        # split out the training and testing data. do the sample for the modulations and snrs
        Xtr, Ytr, Xte, Yte, snrs_te = X[train_index], Y[train_index], X[test_index], Y[test_index], snrs[test_index]

        if scenario == 'A': 
            # sample adversarial training data 
            Ntr = len(Xtr)
            sample_indices = np.random.randint(0, Ntr, Ntr)        

            # train the model
            model_aml = nn_model(X=Xtr[sample_indices], Y=Ytr[sample_indices], train_param=train_adversary_params) 
        
        model = nn_model(X=Xtr, Y=Ytr, train_param=train_params)
        
        if attack_type == 'FastGradientMethod': 
            Xatt = generate_aml_data(model_aml, Xte, Yte, {'type': 'FastGradientMethod', 'eps': 0.15})
        elif attack_type == 'DeepFool': 
            Xatt = generate_aml_data(model_aml, Xte, Yte, {'type': 'DeepFool'})
        elif attack_type == 'ProjectedGradientDescent': 
            Xatt = generate_aml_data(model_aml, Xte, Yte, {'type': 'ProjectedGradientDescent', 
                                                           'eps': 1.0, 
                                                           'eps_step':0.1, 
                                                           'max_iter': 50})
        else: 
            raise(ValueError('Unknown attack type. '))

        # for each of the snrs -> grab all of the data for that snr, which should have all of
        # the classes then evaluate the model on the data for the snr under test. store the 
        # aucs, accs, and ppls in a dictionary 
        for snr in np.unique(snrs_te): 
            Yhat = model.predict(Xte[snrs_te == snr]) 
            Yhat_att = model.predict(Xatt[snrs_te == snr])

            if attack_type == 'FastGradientMethod': 
                result_logger.add_scores(Yte[snrs_te==snr], 
                                     Yhat, Yhat_att, Yhat, Yhat, snr)
            elif attack_type == 'DeepFool': 
                result_logger.add_scores(Yte[snrs_te==snr], 
                                     Yhat, Yhat, Yhat_att, Yhat, snr)
            elif attack_type == 'ProjectedGradientDescent': 
                result_logger.add_scores(Yte[snrs_te==snr], 
                                     Yhat, Yhat, Yhat, Yhat_att, snr)

        # save the results to a pickle file 
        results = {'result_logger': result_logger}
        pickle.dump(results, open(output_path, 'wb'))

        
    result_logger.finalize()

    # save the results to a pickle file 
    results = {'result_logger': result_logger}
    pickle.dump(results, open(output_path, 'wb'))