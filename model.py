
import torch
from torch import nn
from drop_in import LinearWithDropIn


class Model(nn.Module):

    def __init__(self, input_shape: int, layers: list[int], use_drop_in: bool, stop_drop_in_after: int = None, drop_in_kwargs: dict = dict(), activation: any = "Tanh",):

        super().__init__()

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


