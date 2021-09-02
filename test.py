#!/usr/bin/env python 

from arml.utils import load_radioml
from arml.experiments import basic_radioml

# test the basic experiment
file_path = "data/RML2016.10a_dict.pkl"
basic_radioml(file_path=file_path, n_runs=5)

# test data loader 
X, Y, snrs, mods, encoder = load_radioml(file_path=file_path)
print(encoder.inverse_transform(Y[:5]))

dd = 1