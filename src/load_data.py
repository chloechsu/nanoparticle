import numpy as np
import pandas as pd
X_gen = pd.read_csv('data/sim_inputs.csv')
X_sim = pd.read_csv('data/sim_inputs.csv')
y_gen = np.loadtxt('data/gen_outputs.csv', delimiter=',')
y_sim = np.loadtxt('data/sim_outputs.csv', delimiter=',')
print("Data shapes for X_gen, y_gen, X_sim, y_sim:", X_gen.shape, y_gen.shape, X_sim.shape, y_sim.shape)
