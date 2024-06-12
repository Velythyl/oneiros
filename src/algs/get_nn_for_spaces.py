import numpy as np
import torch
from torch import nn


def get_number_of_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters in model:", pytorch_total_params)
    return pytorch_total_params


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def np_prod(inp):
    return int(np.array(inp).prod())


def get_mlp(inp, out, for_actor=True):
    assert len(inp) == len(out) == 1

    ret = nn.Sequential(
        layer_init(nn.Linear(np_prod(inp), 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, np_prod(out)), std=0.01 if for_actor else 1.0),
    )

    return ret


def get_1dcnn_mlp(inp, out, for_actor=True):
    assert len(inp) == 2
    assert len(out) == 1

    class GatedConv1D(torch.nn.Module):
        def __init__(self, in_channels, OBS, out_channels, kernel_size, stride=1, dilation=1, groups=1,
                     activation_func=nn.Tanh):
            super(GatedConv1D, self).__init__()
            self.mask = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          groups=groups),
                nn.Sigmoid()
            )

            self.activation = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          groups=groups),
                activation_func()
            )

        def forward(self, input):
            mask = self.mask(input)
            activation = self.activation(input)

            return mask * activation

    class Conv1DNet(torch.nn.Module):
        def __init__(self, STACK, OBS, OUTPUT_SHAPE, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert isinstance(OUTPUT_SHAPE, int)

            input = torch.randn(100, OBS, STACK)

            activation_func = nn.Tanh

            IN_NETWORK_OUTPUTSHAPE = OUTPUT_SHAPE if for_actor else 8
            IN_NETWORK_BIGSHAPE = 16 if 16 > IN_NETWORK_OUTPUTSHAPE + int(
                0.4 * IN_NETWORK_OUTPUTSHAPE) else IN_NETWORK_OUTPUTSHAPE + int(0.4 * IN_NETWORK_OUTPUTSHAPE)

            def get_list():
                return [
                    GatedConv1D(OBS, OBS, 32, kernel_size=2, dilation=1, activation_func=activation_func),
                    GatedConv1D(32, OBS, 64, kernel_size=2, dilation=2, activation_func=activation_func),
                    GatedConv1D(64, OBS, 32, kernel_size=2, stride=1, dilation=2, activation_func=activation_func),
                ]

            module_list = get_list()

            def call():
                return nn.Sequential(*module_list)(input).shape

            print("Pre-final layer Conv1D policy nets:", call())

            self.hardcoded_backbone = nn.Sequential(*module_list)

            shape = self.hardcoded_backbone(input).shape
            self.final_layer = nn.Sequential(
                GatedConv1D(shape[-2], None, OUTPUT_SHAPE, kernel_size=shape[-1], activation_func=nn.Identity),
                nn.Flatten()
            )

            assert self.final_layer(self.hardcoded_backbone(input)).shape == (100, OUTPUT_SHAPE)

        def forward(self, input):
            input = input.transpose(1, 2)

            conv = self.hardcoded_backbone(input)
            return self.final_layer(conv)

    m = Conv1DNet(inp[0], inp[1], out[0])

    input = torch.randn(500, *inp)
    assert m(input).shape == (500, out[0])

    return m


def get_nets(inp, out, for_actor=True):
    if len(inp) == 1:
        ret = get_mlp(inp, out, for_actor)
    elif len(inp) == 2:
        ret = get_1dcnn_mlp(inp, out, for_actor)
    else:
        raise AssertionError()

    get_number_of_params(ret)
    return ret


if __name__ == "__main__":
    STACK = 10
    OBS = 35
    OUT = 6

    get_nets((STACK, OBS), (OUT,))
    get_nets((OBS,), (OUT,))
