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

from arml.exp import exp_fgsm_impact

# test the basic experiment
file_path = 'data/RML2016.10a_dict.pkl'
# number of cross validation runs 
n_runs = 5
# verbose ? 
verbose = 2
# type of experiment 
scenario = 'A'
# attack epsilons 
epsilons = [0.01, 0.05, 0.1, 0.15,  0.2]
# epsilons = [0.025, 0.75, 0.125, 0.175]
# defenders model 
train_params = {'type': 'vtcnn2', 
                'dropout': 0.5, 
                'val_split': 0.9, 
                'batch_size': 1024, 
                'nb_epoch': 40, 
                'verbose': verbose, 
                'NHWC': [220000, 2, 128, 1],
                'tpu': False, 
                'file_path': 'convmodrecnets_CNN2_0.5.wts.h5'}
# adversary's model 
train_adversary_params = {'type': 'vtcnn2', 
                          'dropout': 0.5, 
                          'val_split': 0.9, 
                          'batch_size': 1024, 
                          'nb_epoch': 40, 
                          'verbose': verbose, 
                          'NHWC': [220000, 2, 128, 1],
                          'tpu': False, 
                          'file_path': 'convmodrecnets_adversary_CNN2_0.5.wts.h5'}
# name for the logger     
logger_name = 'aml_radioml_vtcnn2_vtcnn2_scenario_A'
# output path
output_path = 'outputs/aml_fgsm_vtcnn2_vtcnn2_scenario_A_radioml.pkl'

exp_fgsm_impact(file_path=file_path,
                n_runs=n_runs, 
                verbose=verbose, 
                scenario=scenario,
                epsilons=epsilons, 
                train_params=train_params, 
                train_adversary_params=train_adversary_params, 
                logger_name=logger_name,
                output_path=output_path)


