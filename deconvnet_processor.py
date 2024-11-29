from typing import Tuple

import torch
import torch.nn as nn

from cnn_features_handler import CNNFeaturesHandler
from utils import get_receptive_field_conv2d, get_receptive_field_pool2d


class DeconvnetProcessor(CNNFeaturesHandler):
    def __init__(self, cnn_model_features: list[nn.Module]|nn.Sequential, flip_kernels: bool=False):
        """ New instance initialization """
        super().__init__(cnn_model_features)
        self.deconvnet = self.build_deconvnet_(flip_kernels)
        self._reset()


    def build_deconvnet_(self, flip_kernels: bool=False) -> nn.Module:
        deconvnet = nn.ModuleList()
        for layer in reversed(self.model_features):
            if isinstance(layer, nn.MaxPool2d):
                deconvnet.append(nn.MaxUnpool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                ))
            elif isinstance(layer, nn.ReLU):
                deconvnet.append(nn.ReLU())
            elif isinstance(layer, nn.Conv2d):
                conv = nn.ConvTranspose2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    output_padding=1 if layer.stride[0] > 1 else 0, # Because stride > 1
                    dilation=layer.dilation
                )
                weights = layer.weight
                if flip_kernels:
                    torch.flip(weights, [2, 3])
                conv.weight = weights
                deconvnet.append(conv)

        return deconvnet
    

    def conv_deconv(self, 
                    x: torch.Tensor, 
                    to_layer_idx: int=-1, 
                    kernel_idx: int=0, 
                    flip_kernel: bool=False, 
                    reset_forward_cnn: bool=True,
                    verbose: bool=False) -> torch.Tensor:
        """
        - Arg(s):
            reset_forward_cnn : bool
                if recall with only flip_kernel different
        """
        if reset_forward_cnn or self.forward_outputs_ == None:
            self._reset()
            self.forward_outputs_ = self.get_feature_layer_output(x, to_layer_idx, verbose=verbose)

        y = self.get_cleaned_feature_maps_with_activation(self.forward_outputs_, kernel_idx)
        y = self.forward_deconv(y, to_layer_idx, flip_kernel=flip_kernel, verbose=verbose)
        
        return y
        

    def forward_deconv(self, y: torch.Tensor, from_layer_idx: int=-1, flip_kernel=False, verbose=False):
        """
        Because the deconvnet process is a symetric of the feature part of the CNN model
        """
        from_layer_idx = self.get_normalized_idx(from_layer_idx)
        self.assert_correct_layer_idx(from_layer_idx)

        idx_maxpool_indices = -1
        for i, layer in enumerate(reversed(self.model_features), start=1):
            idx = len(self.model_features) - i

            # only backward from the from_layer_idx'th layer
            if from_layer_idx < idx:
                continue

            if verbose:
                print(f"idx :{idx}", f"y : {y.size()}", layer)

            if isinstance(layer, nn.MaxPool2d):
                indices = self.maxpool_indices[idx_maxpool_indices]
                y = nn.functional.max_unpool2d(
                    y,
                    indices=indices,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                )
                idx_maxpool_indices -= 1
            
            elif isinstance(layer, nn.ReLU):
                y = nn.functional.relu(y)
            
            elif isinstance(layer, nn.Conv2d):
                weights = self.model_features[idx].weight
                if flip_kernel:
                    torch.flip(weights, [2, 3])

                if verbose:
                    print("weights size :", weights.size())

                y = nn.functional.conv_transpose2d(
                    y,
                    weight=weights,
                    #bias=layer.bias,
                    stride=layer.stride,
                    padding=layer.padding,
                    output_padding=1 if layer.stride[0] > 1 else 0, # Because stride > 1
                    dilation=layer.dilation
                )
        
        return y


    def get_cleaned_feature_maps_with_activation(self, feature_maps: torch.tensor, kernel_idx: int=0) -> torch.Tensor:
        """ Returns the zeroed output tensor but the channel_idx'th feature map """
        t = torch.zeros_like(feature_maps)
        strongest_activation_feature_map = self.get_strongest_activation_feature_map(feature_maps[0, kernel_idx])
        t[0, kernel_idx] = strongest_activation_feature_map
        return t
    

    def get_feature_layer_output(self, x: torch.Tensor, to_layer_idx: int=-1, keeping_indices=True, keep_previous: bool=False, verbose=False) -> torch.Tensor:
        """ 
        - Argument(s) :
            x : torch.Tensor
                size = (batch, channels, H, W)
            to_layer_idx : int
            keeping_indices : bool
            verbose : bool 
        """
        to_layer_idx = self.get_normalized_idx(to_layer_idx)
        self.assert_correct_layer_idx(to_layer_idx)
        
        if keeping_indices:
            self.update_model(new_status=True) # Must be done before .val()
            self.model_features.eval()

            outputs = [] if keep_previous else None
            with torch.no_grad():
                self.maxpool_indices.clear()
                for idx, layer in enumerate(self.model_features[:to_layer_idx+1]):
                    print(idx, layer)

                    if isinstance(layer, nn.MaxPool2d):
                        x, indices = layer(x)
                        self.maxpool_indices.append(indices)
                    else:
                        x = layer(x)

                    if keep_previous:
                        outputs.append(x)
            
            self.model_features.train()
            self.update_model(new_status=False)

            return x
        else:
            return super().get_feature_layer_output(x, to_layer_idx, keep_previous=keep_previous, verbose=verbose)
 

    def get_receptive_field_of_activation(self, idx_layer: int, activation_pos: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ """
        idx_layer = self.get_normalized_idx(idx_layer)
        self.assert_correct_layer_idx(idx_layer)
        
        tlc, brc = activation_pos, activation_pos
        for module in reversed(self.model_features[:idx_layer+1]):
            #print(module)
            if isinstance(module, nn.Conv2d):
                tlc, _ = get_receptive_field_conv2d(tlc, module.kernel_size, module.stride, module.padding)
                _, brc = get_receptive_field_conv2d(brc, module.kernel_size, module.stride, module.padding)
            elif isinstance(module, nn.MaxPool2d):
                tlc, _ = get_receptive_field_pool2d(tlc, module.kernel_size, module.stride, module.padding)
                _, brc = get_receptive_field_pool2d(brc, module.kernel_size, module.stride, module.padding)
            #print(tlc, brc)
        return (tlc, brc)


    def get_strongest_activation_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """ 
        """
        o = torch.zeros_like(feature_map)
        idx_max = torch.argmax(feature_map).item()
        row_max = idx_max // feature_map.size(0)
        column_max = idx_max % feature_map.size(1)
        max = feature_map.max()
        self.strongest_activation_ = (max, (row_max, column_max))
        o[row_max, column_max] = self.strongest_activation_[0]
        return o
    

    def _reset(self) -> None:
        """ """
        self.maxpool_indices = []
        self.forward_outputs_ = None
        self.strongest_activation_ = None # will be (float, [int, int])


    def update_model(self, new_status=True) -> None:
        """
            Set MaxPooling layer's return_indices setting to True in order to get indices necessary for deconv backward
        """
        for layer in self.model_features:
            if isinstance(layer, nn.MaxPool2d):
                layer.return_indices=new_status


if __name__ == "__main__":
    # Load a model : AlexNet

    print("No test yet")