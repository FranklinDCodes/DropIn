
import numpy as np
from math import sqrt

SEED = 42

FAN_IN = 256
FAN_OUT = 512

LAYER_VAR = 2 / (FAN_IN + FAN_OUT)
LAYER_MEAN = 0

STARTING_ROW_VAR = 2
STARTING_COL_VAR = 0.1

gen = np.random.Generator(np.random.PCG64())

# create sample row
sample_feat_vec = gen.standard_normal(size=FAN_OUT) * sqrt(STARTING_ROW_VAR)

# generate columns
neur_vecs = list()
for feat_weight in sample_feat_vec:

    # use row weight as mean for col
    # IOW use feature weight as mean for all weights in a given neuron
    neuron_vec = gen.standard_normal(size=FAN_IN) * STARTING_COL_VAR + feat_weight

    neur_vecs.append(neuron_vec)

# stack and transpose to get weight matrix
np_weight_mat_raw = np.stack(neur_vecs).T

# bring back to desired distribution
np_weight_mat = LAYER_MEAN + sqrt(LAYER_VAR / np.var(np_weight_mat_raw)) * (np_weight_mat_raw - np.mean(np_weight_mat_raw))

# check the variance of the whole thing
print(f"Matrix shape: {np_weight_mat.shape}")
print(f"Desired variance: {LAYER_VAR}")
print(f"Matrix variance: {np.var(np_weight_mat)}")
print(f"Mean row variance (feature-to-feature weight variance): {np.mean(np.var(np_weight_mat, 0))}")
print(f"Mean col variance (neuron-to-neuron weight variance): {np.mean(np.var(np_weight_mat, 1))}")





