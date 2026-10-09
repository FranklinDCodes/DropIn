
import torch
from math import sqrt


def Glorot(
        fan_in: int, 
        fan_out: int,
        gen: torch.Generator
    ) -> torch.Tensor:

    return torch.randn((fan_in, fan_out), generator=gen) * (2 / (fan_in + fan_out)) ** 0.5


# code comes from math_development_code/init_test_torch.py
# developed by me and vectorized and torch-ified by Claude Sonnet 5.5
def InterNVG(
        fan_in: int, 
        fan_out: int,
        gen: torch.Generator,
        starting_feature_var: float = 2.0,
        starting_neuron_var: float = 0.1
    ) -> torch.Tensor:

    # create sample row, shape (FAN_OUT,)
    sample_feat_vec = torch.randn(fan_out, generator=gen) * sqrt(starting_feature_var)

    # generate the full weight matrix in one shot, shape (FAN_IN, FAN_OUT)
    # each column j is noise + sample_feat_vec[j] (broadcast across rows)
    weight_mat_raw = torch.randn(fan_in, fan_out, generator=gen) * starting_neuron_var + sample_feat_vec

    glorot_var = sqrt(2 / (fan_in + fan_out))

    # bring back to desired distribution
    weight_mat = torch.sqrt(
    glorot_var / torch.var(weight_mat_raw, unbiased=False)
    ) * (weight_mat_raw - weight_mat_raw.mean())

    return weight_mat


D_INIT_FUNCS = {
    "glorot": Glorot,
    "interNVG": InterNVG
}

def get_init_func(name: str, **kwargs) -> any:

    return lambda fin, fout, gen : D_INIT_FUNCS[name](fin, fout, gen, **kwargs)
