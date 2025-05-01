from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from .clean_map import clean_feature_maps
from .utils_deconv import make_coherent_before_max_unpool2d
from .convnet_wrapper_for_deconvolution import ConvnetWrapperForDeconvolution

__version__ = "1.1.0"

def perform_deconvolution(
        output: torch.Tensor,
        idx_layer: int,
        convnet_features: nn.Module,
        switch_indices: List[Tuple[int, torch.Tensor]],
        flip_kernels: bool = False,
        use_bias: bool = False,
        verbose: bool = False,
        ) -> torch.Tensor:
    """
    
    """
    for i, module in zip(range(idx_layer, -1, -1), reversed(convnet_features[:idx_layer+1])):
        if verbose:
            print(f"[{i}] reverse of ", module)
        
        if isinstance(module, nn.MaxPool2d):
            _, indices = switch_indices.pop()
            output = make_coherent_before_max_unpool2d(output, indices)
            output = nn.functional.max_unpool2d(
                output,
                indices=indices,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding
            )
        elif isinstance(module, nn.ReLU):
            output = nn.functional.relu(output)
        elif isinstance(module, nn.Conv2d):
            weight = torch.flip(module.weight, [2, 3]) if flip_kernels else module.weight
            if use_bias and module.bias != None:
                if verbose:
                    print("\t using bias")
                    #print("bias", output.size(), "module.bias", module.bias.size())
                bias = module.bias.unsqueeze(dim=0).unsqueeze(dim=0).reshape(module.bias.size(0), 1, 1).unsqueeze(dim=0)
                output -= bias
            output = nn.functional.conv_transpose2d(
                output,
                weight=weight,
                stride=module.stride,
                padding=module.padding,
                output_padding=1 if module.stride[0] > 1 else 0, # Because stride > 1
                dilation=module.dilation
            )

        if verbose:
            print(f"\t\t> output.size :{output.size()} | min : {output.min().item():.3f} | max : {output.max().item()}")

    return output


def deconvolution(
        wrapped_convnet: ConvnetWrapperForDeconvolution,
        x: torch.Tensor,
        idx_layer: int = None,
        flip_kernels: bool = False,
        use_bias: bool = False,
        clean_feature_map: bool = True,
        idx_map: Optional[int] = None,
        pos: Optional[Tuple[int, int]] = None,
        return_pos: bool = True,
        verbose: bool = False
        ) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor]:
    """
    Assuming than conv_model have a features composant corresponding to CNN part.

    Args:
        - cnn_model
        - x (torch.Tensor) : input to forward until idx_layer module (size [B, C, H, W])
        - idx_layer
        - flip_kernels
        - use_bias
        - clean_feature_map
        - idx_map (int) : used if clean_feature_map. Chanel index. If False, it returns a value by item. If None, return a max value by channel.
        - pos ((int, int), optional) : used if clean_feature_map. indice of specific activation in any map of feature_maps.
        - return_pos : used if clean_feature_map. return index in feature_maps of kept activations.
        
        - verbose

    Returns:
        - backward deconvolution result
        - coords of max activation, required for receptive field process, if clean_feature_map
    """
    convnet_features = wrapped_convnet.convnet_features

    # Normalizing idx_layer
    if idx_layer == None:
        idx_layer = len(convnet_features) - 1
    elif idx_layer < 0:
        idx_layer += len(convnet_features)
    # check if idx_layer is coherent with cnn_features
    assert (0 <= idx_layer) and (idx_layer < len(convnet_features)), \
        f"i should be in [-{len(convnet_features)}; {len(convnet_features)}["
    
    # Generate feature maps and switch indices
    output, switch_indices = wrapped_convnet.forward_for_deconv(
        x, idx_layer, return_switch_indices=True, verbose=verbose
        )
    
    if verbose:
        print(f"forwarded output size {output.size()} | switch indices : {len(switch_indices)}", end="")
        print(f" | min : {output.min().item()} •| max : {output.max().item()}")

    # Clean idx_map feature maps
    if clean_feature_map:
        pool_indices = None

        if isinstance(convnet_features[idx_layer], nn.MaxPool2d):
            _, pool_indices = switch_indices[-1]
            if verbose:
                print("pool_indices.size", pool_indices.size())

        cleaned = clean_feature_maps(
            output, idx_map=idx_map, pos=pos, pool_indices=pool_indices,
            return_pos=return_pos, keep_only_last_occurrence=False
            )
        
        #if verbose:
        #    print("cleaned", cleaned)

        if return_pos:
            output, pos = cleaned
        else:
            output = cleaned

    # perform deconvnet
    deconv = perform_deconvolution(
        output,
        idx_layer,
        convnet_features,
        switch_indices,
        flip_kernels=flip_kernels,
        use_bias=use_bias,
        verbose=verbose,
        )

    if return_pos and clean_feature_map:
        return deconv, pos
    else:
        return deconv