from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from .convnet_wrapper_for_deconvolution import ConvnetWrapperForDeconvolution
from .utils_deconv import make_coherent_before_max_unpool2d
from .clean_map import clean_feature_maps

__version__ = "1.1.2"

class Sub(nn.Module):
    def __init__(self, tensor_to_sub: torch.Tensor) -> None:
        super().__init__()
        self.tensor_to_sub = tensor_to_sub

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.tensor_to_sub

    def string(self) -> str:
        return f"Sub({self.tensor_to_sub})"


class Deconvnet(nn.Module):
    def __init__(self,
                 wrapped_convnet: ConvnetWrapperForDeconvolution,
                 flip_kernels: bool = False,
                 use_bias: bool = False
                 ) -> None:
        super().__init__()
        self.wrapped_convnet = wrapped_convnet
        self.deconv_model = nn.Sequential()
        self.flip_kernels = flip_kernels
        self.use_bias = use_bias
        self.build_deconvnet()

    def build_deconvnet(self) -> None:
        with torch.no_grad():
            for module in reversed(self.wrapped_convnet.convnet_features):
                if isinstance(module, nn.MaxPool2d):
                    self.deconv_model.append(nn.MaxUnpool2d(
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding
                    ))
                elif isinstance(module, nn.ReLU):
                    self.deconv_model.append(nn.ReLU())
                elif isinstance(module, nn.Conv2d):
                    conv = nn.ConvTranspose2d(
                        in_channels=module.out_channels,
                        out_channels=module.in_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        output_padding=1 if module.stride[0] > 1 else 0, # Because stride > 1
                        dilation=module.dilation,
                        bias=False
                    )
                    weight = module.weight.clone().to("cpu")
                    conv.weight.copy_(torch.flip(weight, [2, 3]) if self.flip_kernels else weight)

                    if self.use_bias and module.bias != None:
                        bias = module.bias.clone().to("cpu").unsqueeze(dim=0).unsqueeze(dim=0).reshape(module.bias.size(0), 1, 1) #.unsqueeze(dim=0)
                        self.deconv_model.append(nn.Sequential(Sub(bias), conv)) # In order to preserve layer mapping with wrapped_convnet features part
                    else:
                        self.deconv_model.append(conv)
                    
    def forward(
            self,
            x: torch.Tensor,
            from_idx_layer: int,
            switch_indices: List[Tuple[int, torch.Tensor]],
            verbose: bool = False
            ) -> torch.Tensor:
        """
        x: input tensor
        switch_indices: list of tuples (index, indices) where index is the index of the layer and indices are the indices to be switched
        """
        if from_idx_layer < 0:
            from_idx_layer += len(self.deconv_model)
        assert from_idx_layer < len(self.deconv_model), f"from_idx_layer must be in range [-{len(self.deconv_model)}, {len(self.deconv_model)}["

        from_idx_layer = len(self.deconv_model) - 1 - from_idx_layer
        if verbose:
                print( f"from_idx_layer : {from_idx_layer}   | len(self.deconv_model) : {len(self.deconv_model)}")
        for i, module in enumerate(self.deconv_model):
            if i < from_idx_layer:
                continue
            if verbose:
                print(f"[{len(self.deconv_model) - 1 - i}] : ", module)
            if isinstance(module, nn.MaxUnpool2d):
                _, indices = switch_indices.pop()
                x = make_coherent_before_max_unpool2d(x, indices)
                x = module(x, indices)
            else:
                x = module(x)
            if verbose:
                print(f"\t\t> output.size :{x.size()} | min : {x.min().item():.3f} | max : {x.max().item()}")
        return x
    
    def deconvolution(
        self,
        x: torch.Tensor,
        idx_layer: int = None,
        clean_feature_map: bool = True,
        idx_map: Optional[int] = None,
        pos: Optional[Tuple[int, int]] = None,
        return_pos: bool = True,
        deconv_device: str = "cpu",
        verbose: bool = False
        ) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor]:
        """
        Assuming than conv_model have a features composant corresponding to CNN part.

        Args:
            - x (torch.Tensor) : input to forward until idx_layer module (size [B, C, H, W])
            - idx_layer (int) : indice of the module from which to get the ouput, if set
            - clean_feature_map (bool) : if True, clean the feature map
            - idx_map (int, optional) : used if clean_feature_map. Chanel index. If False, it returns a value by item. If None, return a max value by channel.
            - pos ((int, int), optional) : used if clean_feature_map. indice of specific activation in any map of feature_maps.
            - return_pos : used if clean_feature_map. return index in feature_maps of kept activations.
            - verbose (bool) : if True, print debug messages

        Returns:
            - backward deconvolution result
            - coords of max activation, required for receptive field process, if clean_feature_map
        """
        convnet_features = self.wrapped_convnet.convnet_features

        # Normalizing idx_layer
        if idx_layer == None:
            idx_layer = len(convnet_features) - 1
        elif idx_layer < 0:
            idx_layer += len(convnet_features)
        # check if idx_layer is coherent with cnn_features
        assert (0 <= idx_layer) and (idx_layer < len(convnet_features)), \
            f"i should be in [-{len(convnet_features)}; {len(convnet_features)}["

        # Generate feature maps and switch indices
        output_to_deconv, switch_indices = self.wrapped_convnet.forward_for_deconv(
            x, idx_layer, return_switch_indices=True, verbose=verbose
            )
        if deconv_device == "cpu":
            output_to_deconv = output_to_deconv.to("cpu").detach()
            switch_indices = [(i, s_i.to("cpu").detach()) for i, s_i in switch_indices]

        if verbose:
            print(f"forwarded output size {output_to_deconv.size()} | switch indices : {len(switch_indices)}", end="")
            print(f" | min : {output_to_deconv.min().item()} | max : {output_to_deconv.max().item()}")

        # Clean idx_map feature maps
        if clean_feature_map:
            pool_indices = None

            # si le dernier module est un pooling, afficher la taille des indices de pooling qu'il a généré
            if isinstance(convnet_features[idx_layer], nn.MaxPool2d):
                _, pool_indices = switch_indices[-1]
                if verbose:
                    print("pool_indices.size", pool_indices.size())

            # Nettoyage de la carte des caractéristiques qu'est la sortie
            cleaned = clean_feature_maps(
                output_to_deconv, idx_map=idx_map, pos=pos, pool_indices=pool_indices,
                return_pos=return_pos, keep_only_last_occurrence=False
                )

            if return_pos:
                output_to_deconv, pos = cleaned
            else:
                output_to_deconv = cleaned

        # perform deconvnet
        deconv = self.forward(
            output_to_deconv,
            idx_layer,
            switch_indices,
            verbose=verbose
        )

        if return_pos and clean_feature_map:
            return deconv, pos
        else:
            return deconv