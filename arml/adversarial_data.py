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
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent


def generate_aml_data(model, X:np.ndarry, Y:np.ndarray, attack_params:dict):
    """generate the adversarial evasion data 
    """

    classifier = KerasClassifier(model=model, clip_values=(-5.0, 5.0), use_logits=False)

    if attack_params['type'] == 'FastGradientMethod': 
        attack = FastGradientMethod(estimator=classifier, eps=attack_params['eps'])
    elif attack_params['type'] == 'DeepFool': 
        attack = DeepFool(classifier, verbose=False)
    elif attack_params['type'] == 'ProjectedGradientDescent': 
        attack = ProjectedGradientDescent(classifier, eps=attack_params['eps'], eps_step=attack_params['eps_step'], verbose=False)
    else: 
        raise(ValueError(''.join(['Unknown attack ', attack_params['type']])))

    Xadv = attack.generate(x=X)
    return Xadv