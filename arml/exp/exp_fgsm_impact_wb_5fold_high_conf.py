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
from ..performance import FGSMPerfLogger
from ..adversarial_data import generate_aml_data
from art.defences.postprocessor import HighConfidence

from sklearn.model_selection import KFold

def experiment_fgsm_wb_5fold_high_conf(file_path:str,
                    n_runs:int=5, 
                    verbose:int=1, 
                    # scenario:str='A',
                    epsilons:list=[0.00025, 0.0005],  # [0.25, 0.75, 0.125, 0.175]
                    train_params:dict={}, 
                    # train_adversary_params:dict={}, 
                    logger_name_1:str='vtcnn2_FGSM_1fold_wb',
                    output_path_1:str='outputs/vtcnn2_FGSM_wb_1fold_op.pkl',
                    logger_name_2:str='vtcnn2_FGSM_1fold_wb_high_conf',
                    output_path_2:str='outputs/vtcnn2_FGSM_wb_1fold_high_conf_op.pkl'
                    ): 
    """evaluate different values of epsilon with FGSM

    Robustness and Inter-Architecture Portability of Deep Neural Networks 
    Parameters
    ---------- 
    file_path : str
        Location of the radioml dataset
    n_runs : int
        Number of cross validations  
    verbose : int
        Verbose?  
    scenario : str 
        Adversary knowledge: 
            'A': has an NN structure and a subset of the training data  
    epsilons : list 
        List of adversarial budgets 
    train_params : dict
        Training parameters
            train_params = {'type': 'vtcnn2', 
                        'dropout': 0.5, 
                        'val_split': 0.9, 
                        'batch_size': 1024, 
                        'nb_epoch': 50, 
                        'verbose': verbose, 
                        'NHWC': [N, H, W, C],
                        'tpu': False, 
                        'file_path': 'convmodrecnets_CNN2_0.5.wts.h5'}
    train_adversary_params : dict
        Training parameters 
            train_adversary_params = {'type': 'vtcnn2', 
                                  'dropout': 0.5, 
                                  'val_split': 0.9, 
                                  'batch_size': 1024, 
                                  'nb_epoch': 50, 
                                  'verbose': verbose, 
                                  'NHWC': [N, H, W, C],
                                  'epsilon': 0.15, 
                                  'file_path': 'convmodrecnets_adversary_CNN2_0.5.wts.h5'}
    logger_name : str
        Name of the logger class [default: 'aml_radioml_vtcnn2_vtcnn2_scenario_A']
    output_path : str
        Output path [default: outputs/aml_vtcnn2_vtcnn2_scenario_A_radioml.pkl]
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
                        'file_path': 'FGSM_CNN2_5fold.wts.h5'}
    
    # if len(train_adversary_params) == 0:
    #     train_adversary_params = {'type': 'vtcnn2', 
    #                               'dropout': 0.5, 
    #                               'val_split': 0.9, 
    #                               'batch_size': 1024, 
    #                               'nb_epoch': 50, 
    #                               'verbose': verbose, 
    #                               'NHWC': [N, H, W, C],
    #                               'tpu': False, 
    #                               'file_path': 'convmodrecnets_adversary_CNN2_0.5.wts.h5'}
    
    # initialize the performances to empty 
    result_logger = FGSMPerfLogger(name=logger_name_1, 
                                   snrs=np.unique(snrs), 
                                   mods=np.unique(mods), 
                                   params=[train_params, train_params], 
                                   epsilons=epsilons)

    result_logger_high_conf = FGSMPerfLogger(name=logger_name_2, 
                                   snrs=np.unique(snrs), 
                                   mods=np.unique(mods), 
                                   params=[train_params, train_params], 
                                   epsilons=epsilons)
    kf = KFold(n_splits=n_runs)
    
    for train_index, test_index in kf.split(X): 
        # split out the training and testing data. do the sample for the modulations and snrs
        Xtr, Ytr, Xte, Yte, snrs_te = X[train_index], Y[train_index], X[test_index], Y[test_index], snrs[test_index]

        # if scenario == 'A': 
        #     # sample adversarial training data 
        #     Ntr = len(Xtr)
        #     sample_indices = np.random.randint(0, Ntr, Ntr)        

            # train the model
            # model_aml = nn_model(X=Xtr[sample_indices], Y=Ytr[sample_indices], train_param=train_adversary_params) 
        
        model = nn_model(X=Xtr, Y=Ytr, train_param=train_params)
        model_aml = model
        
        # loop through the different values of epsilon and generate adversarial datasets
        for eps_index, epsilon in enumerate(epsilons): 
            Xfgsm = generate_aml_data(model_aml, Xte, Yte, {'type': 'FastGradientMethod', 'eps': epsilon})

            for snr in np.unique(snrs_te): 
                Yhat_fgsm = model.predict(Xfgsm[snrs_te == snr])
                result_logger.add_scores(Yte[snrs_te==snr], Yhat_fgsm, snr, eps_index)
                postprocessor = HighConfidence(cutoff=0.3)
                post_preds = postprocessor(preds=Yhat_fgsm)
                result_logger_high_conf.add_scores(Yte[snrs_te==snr], post_preds, snr, eps_index)
            
        result_logger.increment_count()
        result_logger_high_conf.increment_count()

        # save the results to a pickle file 
        results = {'result_logger': result_logger}
        pickle.dump(results, open(output_path_1, 'wb'))
        results_high_conf = {'result_logger_high_conf': result_logger_high_conf}
        pickle.dump(results_high_conf, open(output_path_2, 'wb'))

        
    result_logger.finalize()
    result_logger_high_conf.finalize()
    # save the results to a pickle file 
    results = {'result_logger': result_logger}
    pickle.dump(results, open(output_path_1, 'wb'))

    results_high_conf = {'result_logger_high_conf': result_logger_high_conf}
    pickle.dump(results_high_conf, open(output_path_2, 'wb'))
