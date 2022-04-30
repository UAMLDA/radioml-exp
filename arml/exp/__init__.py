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

from .exp_basic import experiment_basic_radioml as exp_basic
from .exp_fgsm_impact import experiment_fgsm as exp_fgsm_impact
from .exp_fgsm_impact_wb_5fold import experiment_fgsm_wb_5fold as exp_fgsm_impact_wb_5fold
from .exp_fgsm_impact_wb_1fold import experiment_fgsm_wb_1fold as exp_fgsm_impact_wb_1fold
from .exp_fgsm_impact_wb_1fold_high_conf import experiment_fgsm_wb_1fold_high_conf as exp_fgsm_impact_wb_1fold_high_conf
from .exp_fgsm_impact_wb_1fold_shift import experiment_fgsm_wb_1fold_shift as exp_fgsm_impact_wb_1fold_shift
from .exp_fgsm_impact_wb_5fold_shift import experiment_fgsm_wb_5fold_shift as exp_fgsm_impact_wb_5fold_shift
from .exp_fgsm_impact_wb_1fold_shift1 import experiment_fgsm_wb_1fold_shift1 as exp_fgsm_impact_wb_1fold_shift1
from .exp_fgsm_impact_wb_5fold_shift1 import experiment_fgsm_wb_5fold_shift1 as exp_fgsm_impact_wb_5fold_shift1
from .exp_fgsm_impact_wb_1fold_shift5 import experiment_fgsm_wb_1fold_shift5 as exp_fgsm_impact_wb_1fold_shift5
from .exp_fgsm_impact_wb_1fold_shift10 import experiment_fgsm_wb_1fold_shift10 as exp_fgsm_impact_wb_1fold_shift10
from .exp_multiple_attack import experiment_adversarial as exp_multiple_attack
from .exp_single_attack import experiment_single_adversarial as exp_single_attack

__all__ = [
    'exp_basic', 
    'exp_fgsm_impact', 
    'exp_fgsm_impact_wb_5fold',
    'exp_fgsm_impact_wb_1fold',
    'exp_fgsm_impact_wb_1fold_high_conf',
    'exp_fgsm_impact_wb_1fold_shift',
    'exp_fgsm_impact_wb_5fold_shift',
    'exp_fgsm_impact_wb_1fold_shift1',
    'exp_fgsm_impact_wb_5fold_shift1',
    'exp_fgsm_impact_wb_1fold_shift5',
    'exp_fgsm_impact_wb_1fold_shift10',
    'exp_multiple_attack', 
    'exp_single_attack'
]