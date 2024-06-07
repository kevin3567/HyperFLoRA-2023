import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.modules.a_abstracts import ComponentAbstract


class FCL(ComponentAbstract):
    def __init__(self, layer_params, activation_type):
        super().__init__()

        if activation_type.upper() == "RELU":
            create_activation = nn.ReLU
        else:
            assert False, "For FCL (Inferer), activation type \"{}\" does not exist.".format(activation_type)
        _seq_layers = []
        for l_idx, (in_size, out_size) in enumerate(layer_params):
            _seq_layers.append(nn.Linear(in_size, out_size))
            if l_idx < (len(layer_params) - 1):
                _seq_layers.append(create_activation())  # no activation on the final layer
        self.seq_layer = nn.Sequential(*_seq_layers)  # alternate between weight and activation

    def forward(self, x):
        y = self.seq_layer(x)
        return y

    def init_param(self):
        for layer in self.seq_layer:
            if hasattr(layer, "weight") and layer.weight.dim() > 1:
                nn.init.xavier_normal_(layer.weight)

