
import torch
from torch import nn
from drop_in import LinearWithDropIn
from param_init import get_init_func


class Model(nn.Module):

    def __init__(self, input_shape: int, layers: list[int], use_drop_in: bool, stop_drop_in_after: int = None, drop_in_kwargs: dict = dict(), activation: any = "Tanh"):

        super().__init__()


        # BUILD LAYERS

        d_activations = {
            "Tanh": nn.Tanh,
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU
        }

        self.stop_drop_in_after = stop_drop_in_after

        # build layers
        l_layers = list()
        fan_in = input_shape
        for idx, layer_size in enumerate(layers):

            if use_drop_in:
                layer = LinearWithDropIn(fan_in, layer_size, **drop_in_kwargs)
            else:
                layer = nn.Linear(fan_in, layer_size)

            # add layer
            fan_in = layer_size
            l_layers.append(layer)

            # add activation
            if idx < len(layers) - 1:
                l_layers.append(d_activations[activation]())

        self.layers = nn.Sequential(*l_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)

    # for stopping 
    def update(self, epoch: int) -> None:

        # if there is a setting to turn off drop in and it's triggered
        if self.stop_drop_in_after and epoch >= self.stop_drop_in_after:
            for layer in self.layers:

                if isinstance(layer, LinearWithDropIn):
                    layer.turn_off_drop_in()

class ParamInitializer:

    def __init__(self, seed: any, init_func_name: str, **kwargs):

        # grab init func
        self._init_func = get_init_func(init_func_name, **kwargs)

        # create generator
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    # init a single linear layer
    # takes an init func that takes a fan_in, fan_out and outputs a mat fan_in X fan_out
    def _init_linear(self, layer: nn.Linear, init_fn: any, gen: torch.Generator):

        fan_in, fan_out = layer.in_features, layer.out_features
        W = init_fn(fan_in, fan_out, gen)
        with torch.no_grad():
            layer.weight.copy_(W.T) # nn.Linear stores (out, in), so transpose
            if layer.bias is not None:
                layer.bias.zero_()

    # looks for model with .layers attribute
    def initialize_layers(self, model: nn.Module) -> None:

        for layer in model.layers:

            # check layer type
            if isinstance(layer, nn.Linear):
                self._init_linear(layer, self._init_func, self.gen)

    def __call__(self, model: nn.Module) -> None:

        return self.initialize_layers(model)

