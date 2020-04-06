import numpy as np
import pandas as pd

import glob

def read_split_files(base_filepath, is_pandas=False):
    if base_filepath.endswith('.csv'):
        base_filepath = base_filepath[:-4]
    total_rows = 0
    for f in glob.glob(base_filepath+'*.csv'):
        if is_pandas:
            total_rows += pd.read_csv(f).shape[0]
        else:
            total_rows += np.loadtxt(f, delimiter=',').shape[0]
    print('Parsed %d rows from %s.' % (total_rows, base_filepath))

print('Parsing simulated training data..')
X_gen = read_split_files('data/gen_geom_spectral.csv', is_pandas=True)
X_sim_train = read_split_files('data/sim_train_geom_spectral.csv', is_pandas=True)
X_sim_test = read_split_files('data/sim_test_geom_spectral.csv', is_pandas=True)
y_gen = read_split_files('data/gen_emi_spectral.csv')
y_sim_train = read_split_files('data/sim_train_emi_spectral.csv')
y_sim_test = read_split_files('data/sim_test_emi_spectral.csv')
