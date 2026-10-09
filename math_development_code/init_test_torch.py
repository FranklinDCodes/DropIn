
"""

    This is the code I wrote in init_test refactored an PyTorch-ified by Claude Sonnet 5.5 Medium

"""

import torch
from math import sqrt

SEED = 42

FAN_IN = 256
FAN_OUT = 512

LAYER_VAR = 2 / (FAN_IN + FAN_OUT)
LAYER_MEAN = 0

STARTING_ROW_VAR = 2
STARTING_COL_VAR = 0.1

gen = torch.Generator().manual_seed(SEED)

# create sample row, shape (FAN_OUT,)
sample_feat_vec = torch.randn(FAN_OUT, generator=gen) * sqrt(STARTING_ROW_VAR)

# generate the full weight matrix in one shot, shape (FAN_IN, FAN_OUT)
# each column j is noise + sample_feat_vec[j] (broadcast across rows)
weight_mat_raw = torch.randn(FAN_IN, FAN_OUT, generator=gen) * STARTING_COL_VAR + sample_feat_vec

# bring back to desired distribution
weight_mat = LAYER_MEAN + torch.sqrt(
    LAYER_VAR / torch.var(weight_mat_raw, unbiased=False)
) * (weight_mat_raw - weight_mat_raw.mean())


# check the variance of the whole thing
print(f"Matrix shape: {weight_mat.shape}")
print(f"Desired variance: {LAYER_VAR}")
print(f"Matrix variance: {torch.var(weight_mat, unbiased=False)}")
print(f"Mean row variance (feature-to-feature weight variance): {torch.mean(torch.var(weight_mat, 0, unbiased=False))}")
print(f"Mean col variance (neuron-to-neuron weight variance): {torch.mean(torch.var(weight_mat, 1, unbiased=False))}")


