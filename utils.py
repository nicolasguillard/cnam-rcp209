import math
from typing import Tuple
import torch.nn as nn

__version__ = "1.0.2"

def get_conv_params(conv: nn.Conv2d) -> dict:
    """
    (FR) Récupération des principaux paramètres d'une instance Conv2d sous forme
    de dictionnaire compatible avec les noms officiels des arguments pour la
    création d'une instance.
    """
    get_keys = [
        "in_channels", "out_channels", "kernel_size", "stride", "padding", "padding_mode", "dilation"
        ]
    values = vars(conv)
    params = {k:(values[k] if k in values else None) for k in get_keys}
    params["bias"] = conv.bias is not None
    return params


def get_maxpool_params(conv: nn.MaxPool2d) -> dict:
    """ 
    (FR) Récupération des principaux paramètres d'une instance MaxPool2d sous forme
    de dictionnaire compatible avec les noms officiels des arguments pour la
    création d'une instance.
    """
    get_keys = [
        "kernel_size", "stride", "padding", "dilation"
        ]
    values = vars(conv)
    params = {k:(values[k] if k in values else None) for k in get_keys}
    return params


def conv2d_output_size(
        input_size: int|Tuple[int, int],
        kernel_size: int|Tuple[int, int],
        padding_size: int|Tuple[int, int]=0,
        stride_size: int|Tuple[int, int]=1,
        dilation_size: int|Tuple[int, int]=1,
        pooling_size: int|Tuple[int, int]=1
        ):
    """
    (FR) Calcul des dimensions de la sortie d'un conv2d
    d'après les équations fournies dans https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    _to_tuple = lambda v: (v, v)
    
    if type(input_size) == int:
        input_size = _to_tuple(input_size)
    elif type(input_size) == dict:
        input_size = (input_size["height"], input_size["width"])
    
    if type(kernel_size) == int:
        kernel_size = _to_tuple(kernel_size)
    
    if type(padding_size) == int:
        padding_size = _to_tuple(padding_size)

    if type(stride_size) == int:
        stride_size = _to_tuple(stride_size)
    
    if type(dilation_size) == int:
        dilation_size = _to_tuple(dilation_size)
    
    if type(pooling_size) == int:
        pooling_size = _to_tuple(pooling_size)

    output_size = lambda i: math.floor((
         (input_size[i] + 2 * padding_size[i] - dilation_size[i] * (kernel_size[i] - 1) - 1) / stride_size[i] + 1) / pooling_size[i]
         )
    output_size_h = output_size(0)
    output_size_w = output_size(1)
    return {"height": output_size_h, "width": output_size_w}


def maxpool2d_output_size(
        input_size: int|Tuple[int, int],
        kernel_size: int|Tuple[int, int],
        padding_size: int|Tuple[int, int]=0,
        stride_size: int|Tuple[int, int]=1,
        dilation_size: int|Tuple[int, int]=1,
        ):
    """
    (FR) Calcul des dimensions de la sortie d'un maxpool2d
        d'après les équations fournies dans https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    """
    _to_tuple = lambda v: (v, v)

    if type(input_size) == int:
        input_size = _to_tuple(input_size)
    elif type(input_size) == dict:
        input_size = (input_size["height"], input_size["width"])
    
    if type(kernel_size) == int:
        kernel_size = _to_tuple(kernel_size)
    
    if type(padding_size) == int:
        padding_size = _to_tuple(padding_size)

    if type(stride_size) == int:
        stride_size = _to_tuple(stride_size)
    
    if type(dilation_size) == int:
        dilation_size = _to_tuple(dilation_size)

    output_size = lambda i: math.floor((
         (input_size[i] + 2 * padding_size[i] - dilation_size[i] * (kernel_size[i] - 1) - 1) / stride_size[i] + 1)
         )
    output_size_h = output_size(0)
    output_size_w = output_size(1)
    return {"height": output_size_h, "width": output_size_w}

    
def get_receptive_field(pos, k, s, p):
    """
    (FR) Retoure les coordonnées (row, col) des coins "min-min" et "max-max"
    du champ réceptif ...
    """
    _to_tuple = lambda v: (v, v)
    if type(k) == int:
        k = _to_tuple(k)
    if type(s) == int:
        s = _to_tuple(s)
    if type(p) == int:
        p = _to_tuple(p)
    column = -p[1] + pos[1] * s[1]
    row = -p[0] + pos[0] * s[0]
    return (row, column), (row+k[1], column+k[0])

def get_receptive_field_conv2d(pos, k, s, p):
    """
    (FR) Retoure les coordonnées (row, col) des coins "min-min" et "max-max"
    du champ réceptif ...
    """
    return get_receptive_field(pos, k, s, p)


def get_receptive_field_pool2d(pos, k, s, p):
    """
    (FR) Retoure les coordonnées (row, col) des coins "min-min" et "max-max"
    du champ réceptif ...
    """
    return get_receptive_field(pos, k, s, p)


if __name__ == "__main__":
    get_tuple = lambda v : (v, v)

    # Testing get_conv_params
    def testing_get_conv_params():
        print("** Testing get_conv_params **")

        ## 1
        args = {
            "in_channels": 3, 
            "out_channels": 64,
            "kernel_size": 7,
            "padding": 2,
            "stride": 3,
            "padding_mode": "reflect",
            "dilation": 2,
            "bias": False
        }

        conv2d = nn.Conv2d(**args)
        conv_params = get_conv_params(conv2d)
        print(conv_params)
        args2 = {
            "in_channels": args["in_channels"], 
            "out_channels": args["out_channels"],
            "kernel_size": get_tuple(args["kernel_size"]),
            "padding": get_tuple(args["padding"]),
            "stride": get_tuple(args["stride"]),
            "padding_mode": args["padding_mode"],
            "dilation": get_tuple(args["dilation"]),
            "bias": False
        }
        print(args2)
        assert args2 == conv_params, "Issue with get_conv_params"

        ## 2
        args = {
            "in_channels": 4, 
            "out_channels": 64,
            "kernel_size": 7,
            "padding": 0,
            "stride": 3,
            "padding_mode": "zeros",
            "dilation": 2,
            "bias": True
        }

        conv2d = nn.Conv2d(**args)
        conv_params = get_conv_params(conv2d)
        print(conv_params)
        args2 = {
            "in_channels": args["in_channels"], 
            "out_channels": args["out_channels"],
            "kernel_size": get_tuple(args["kernel_size"]),
            "padding": get_tuple(args["padding"]),
            "stride": get_tuple(args["stride"]),
            "padding_mode": args["padding_mode"],
            "dilation": get_tuple(args["dilation"]),
            "bias": True
        }
        print(args2)
        assert args2 == conv_params, "Issue with get_conv_params"

        print("\t ** All ok **")
    
    testing_get_conv_params()


    # Testing conv2d_output_size
    def testing_conv2d_output_size():
        print("** Testing conv2d_output_size **")

        output_size = conv2d_output_size(
            input_size=(224, 224),
            kernel_size=11,
            padding_size=2,
            stride_size=4,
            dilation_size=1,
            pooling_size=1
        )
        expected = {"height": 55, "width": 55}
        print(output_size)
        assert output_size == expected, "Issue with conv2d_output_size"

        output_size = conv2d_output_size(
            input_size=(27, 27),
            kernel_size=5,
            padding_size=2,
            stride_size=1,
            dilation_size=1,
            pooling_size=2
        )
        expected = {"height": 13, "width": 13}
        print(output_size)
        assert output_size == expected, "Issue with conv2d_output_size"

        print("\t ** All ok **")

    testing_conv2d_output_size()

    ##TODO Testing get_receptive_field_conv2d
    ##TODO Testing get_receptive_field_pool2d