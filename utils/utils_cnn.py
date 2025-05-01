import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torchinfo


__version__ = "1.2.1"

# Also used in PyTorch codes
_pair = lambda v: (v, v)

#################
### CNN utilities
#################

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

    
def get_receptive_field_2d(
        pos: Tuple[int, int],
        kernel: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        input_size: Optional[Tuple[int, int]] = None
        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    (FR) Calcul le champ receptif d'un neurone de position pos=(x, y) dans l'espace de sortie d'une transformation, connaissant ses caractéristiques (conv2d, pool2d). En base 0

    Assuming dilation = 1

    Args:
        pos (int, int): position dans l'espace de sortie (R, C)
        kernel (int | int,int): dimention du noyau
        stide (int | int,int): pas d'avancement
        padding (int | int,int): marge prise en compte dans l'espace de départ
        input_size (Tuple[float, float], optional): size of the input, assuming shape (H, W)

    Returns:
        (Rmin, Cmin), (Rmax, Cmax)
    """
    if type(kernel) == int:
        kernel = _pair(kernel)
    if type(stride) == int:
        stride = _pair(stride)
    if type(padding) == int:
        padding = _pair(padding)
    row = -padding[0] + pos[0] * stride[0]
    column = -padding[1] + pos[1] * stride[1]
    
    if input_size != None:
        return (max(row, 0), max(column, 0)), \
            (min(row+kernel[1]-1, input_size[1]-1), min(column+kernel[0]-1, input_size[0]-1))
    else:
        return (row, column), (row+kernel[1]-1, column+kernel[0]-1)


def get_receptive_field_conv2d(
        pos: Tuple[int, int],
        kernel: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        input_size: Optional[Tuple[int, int]] = None
        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return get_receptive_field_2d(pos, kernel, stride, padding, input_size)


def get_receptive_field_pool2d(
        pos: Tuple[int, int],
        kernel: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        input_size: Optional[Tuple[int, int]] = None
        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return get_receptive_field_2d(pos, kernel, stride, padding, input_size)


def get_receptive_field_in_pixel_space(
        pos: Tuple[int, int],
        idx_layer: int,
        cnn_modules: nn.Module,
        output_sizes: List[Tuple[int, int]],
        pixel_space_size: Tuple[int, int],
        verbose=False
        ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Args:
        - pos ((int, int)): position in the channel in target activation (no need of channel index), shaped (R, C), base 0
        - idx_layer (int): base 0 index of the targeted layer (output of a module)
        - cnn_modules (nn.Module): list of modules in forward order
        - output_sizes (list[(int, int)]): list of output sizes of each respective module
        - pixel_space_size ((int, int)): size of the pixel space

    Returns:
        rc_min, rc_max : ((int, int), (int, int))
    """
    if idx_layer < 0:
        idx_layer += len(cnn_modules)
    assert 0 <= idx_layer and idx_layer < len(cnn_modules), \
        f"idx_layer ({idx_layer}) should be in [-{len(cnn_modules)}; {len(cnn_modules)-1}]"

    output_size = output_sizes[idx_layer]
    if pos[0] < 0:
        pos[0] += output_size[0]
    if pos[1] < 0:
        pos[1] += output_size[1]
    assert 0 <= pos[0] and  pos[0] < output_size[0] and 0 <= pos[1] and  pos[1] < output_size[1], \
        f"pos {pos} not coherent with output size {output_size} of module #{idx_layer} {cnn_modules[idx_layer]}"

    rc_min = rc_max = pos
    modules = list(reversed(cnn_modules[:idx_layer+1]))
    for idx, module in zip(range(len(modules)-1, -1, -1), modules):
        if verbose:
            print(idx, module) ##DEBUG
        
        if idx > 0:
            input_size = output_sizes[idx - 1]
        else:
            input_size = pixel_space_size
        
        if isinstance(module, nn.Conv2d):
            rc_min, _ = get_receptive_field_conv2d(
                rc_min, module.kernel_size, module.stride, module.padding, input_size
                )
            _, rc_max = get_receptive_field_conv2d(
                rc_max, module.kernel_size, module.stride, module.padding, input_size
                )
        elif isinstance(module, nn.MaxPool2d):
            rc_min, _ = get_receptive_field_pool2d(
                rc_min, module.kernel_size, module.stride, module.padding, input_size
                )
            _, rc_max = get_receptive_field_pool2d(
                rc_max, module.kernel_size, module.stride, module.padding, input_size
                )
    
    return rc_min, rc_max


def get_output_sizes(
        model: nn.Module,
        input_size: torch.Size,
        last_2d: bool = True,
        verbose: bool = False
        ) -> List[Tuple[int , int]] :
    stats = torchinfo.summary(model, input_size=input_size, mode="eval")
    if verbose:
        print(stats)
    if last_2d:
        output_sizes = [
            (summary.output_size[-2], summary.output_size[-1]) for summary in stats.summary_list[1:]
            ]
    else:
        output_sizes = [
            (summary.output_size[-3], summary.output_size[-2], summary.output_size[-1]) for summary in stats.summary_list[1:]
            ]
    return output_sizes


if __name__ == "__main__":
    get_tuple = lambda v : (v, v)

    #----------------------
    # Testing CNN utilities
    #----------------------

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