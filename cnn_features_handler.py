import torch
import torch.nn as nn

from utils import get_conv_params, get_maxpool_params, conv2d_output_size, maxpool2d_output_size

from torchinfo import summary


class CNNFeaturesHandler():
    """ """
    def __init__(self, model_features: list[nn.Module]|nn.Sequential):
        """ New instance initialization """
        # nn.ModuleList(model_features)
        if not isinstance(model_features, nn.Sequential):
            self.model_features = nn.Sequential(model_features)
        else:
            self.model_features = model_features
        self.model_summary = None

    def assert_correct_layer_idx(self, idx: int) -> None:
        assert self.is_correct_idx(idx), f"Layer index (idx = {idx}) must be in range [0, {len(self.model_features)})"


    def get_conv_layers_count(self):
        """
        Returns the number of conv layer in the feature part of the model
        """
        return len(self.get_conv_indexes())


    def get_conv_filters(self, conv_pos: int) -> torch.Tensor:
        """
        Returns filters parameters a.k.a weights of the conv layer at pos conv_pos in the features part of the model
        """
        conv = self.get_conv_layer(conv_pos)

        return conv.weight


    def get_conv_layer(self, conv_pos: int) -> nn.Conv2d:
        """
        Returns the conv layer at pos conv_pos in the features part of the model
        """
        index = self.get_conv_index(conv_pos)

        return self.get_feature_layer(index)
    

    def get_conv_index(self, conv_pos: int) -> int:
        """ 
        - Argument(s)
            i : int
                i-th conv layer position order in model feature layer list
        - Return(s)
            idx : int
                index of conv layer position in model feature layer list
        """
        conv_indexes = self.get_conv_indexes()
        len_conv_indexes = len(conv_indexes)
        assert conv_pos in range(1, len_conv_indexes + 1), f"Layer position (conv_pos = {conv_pos}) must be in range [1, {len_conv_indexes}]"

        return conv_indexes[conv_pos-1]


    def get_conv_indexes(self):
        """
        Returns list of indexes of each conv layer in the features part of the model
        """
        return [idx for idx, layer in enumerate(self.model_features) if isinstance(layer, nn.Conv2d)]


    def get_conv_output(self, x: torch.Tensor, conv_pos: int, keep_previous=False) -> torch.Tensor:
        """
        - Argument(s)
            conv_pos : int
                Position de la couche entre 1 et le nombre de couches de conv (pas l'indice dans l'ensemble des couches de la partie features du modele)
        """
        idx = self.get_conv_index(conv_pos)

        return self.get_feature_layer_output(x, idx, keep_previous=keep_previous)


    def get_feature_layer(self, idx: int) -> nn.Module:
        """

        """
        idx = self.get_normalized_idx(idx)
        self.assert_correct_layer_idx(idx)

        return self.model_features[idx]
    

    def get_feature_layer_output(self, x: torch.Tensor, layer_idx: int=-1, keep_previous: bool=False, verbose: bool=False) -> torch.Tensor:
        """
        Returns output of a specific layer of feature part of the model, and each previous layer output if required

        - Argument(s):
            x : torch.Tensor
                input data
            idx : int
                idex of the layer. -1 : last one
            keep_previous : bool
                list of output each layer, if required (default: False)
        - Return(s)
            x : torch.Tensor
                stack output (last features layer stack output)
            outputs : None|[torch.Tensor]
                If required, list of each layer output
        """
        layer_idx = self.get_normalized_idx(layer_idx)
        self.assert_correct_layer_idx(layer_idx)

        outputs = [] if keep_previous else None
        with torch.no_grad():
            for idx, layer in enumerate(self.model_features):
                layer.eval()
                x = layer(x)
                if keep_previous:
                    outputs.append(x)
                if idx == layer_idx:
                    break;
        
            if verbose:
                print(x.size(), f"after {idx}", layer)
        return x, outputs

    
    def get_feature_layer_summary(self,  idx: int=-1, raise_exception: bool=True):
        """
        """
        if not self.model_summary:
            msg = "Process summary first calling '.summary()'"
            if raise_exception:
                raise IndexError(msg)
            else:
                print(msg)
                return None
        
        idx = self.get_normalized_idx(idx)
        self.assert_correct_layer_idx(idx)

        ## Because of nn.Sequential is the first of summary
        if isinstance(self.model_features, nn.Sequential):
            idx = idx + 1

        return self.model_summary.summary_list[idx].output_size


    def get_features_output(self, x, keep_previous=False):
        """
        Returns output of features part of the model , and each layer output if required
        
        - Argument(s):
            x : torch.Tensor
                input data
            keep_previous : bool
                list of output each layer, if required (default: False)
        - Return(s)
            x : torch.Tensor
                stack output (last features layer stack output)
            outputs : None|[torch.Tensor]
                If required, list of each layer output
        """
        return self.get_feature_layer_output(x, keep_previous=keep_previous)
    

    def get_layer_def(self, layer_idx: int) -> dict:
        """ 
        Returns the layer corresponding to the conv layer at pos conv_pos in the features part of the model
        """
        layer = self.get_feature_layer(layer_idx)
        if isinstance(layer, nn.Conv2d):
            return get_conv_params(layer)
        if isinstance(layer, nn.MaxPool2d):
            return get_maxpool_params(layer)

        return None
    

    def get_normalized_idx(self, layer_idx: int=-1):
        """ """
        if layer_idx < 0:
            layer_idx = len(self.model_features) + layer_idx
        
        return layer_idx


    def is_correct_idx(self, idx: int) -> bool:
        """
        Check if idx is in the range of the feature part of the model
        """
        return idx in range(len(self.model_features))


    def show_conv_list(self, input_size=(100, 100)):
        """ """
        pos = 1
        output_size = input_size
        for idx, layer in enumerate(self.model_features):
            args = self.get_layer_def(idx)
            if isinstance(layer, nn.Conv2d):
                
                output_size = conv2d_output_size(
                    input_size=output_size,
                    kernel_size=args["kernel_size"],
                    padding_size=args["padding"],
                    stride_size=args["stride"],
                    dilation_size=args["dilation"]
                )
                print(f"pos: {pos:3d} - idx: {idx:3d} -", self.get_conv_layer(pos), f"- output size : {output_size}")
                pos += 1
            elif isinstance(layer, nn.MaxPool2d):
                output_size = maxpool2d_output_size(
                    input_size=output_size,
                    kernel_size=args["kernel_size"],
                    padding_size=args["padding"],
                    stride_size=args["stride"],
                    dilation_size=args["dilation"]
                )


    def show_feature_list(self):
        """ """
        for idx, layer in enumerate(self.model_features):
            print(f"idx: {idx:3d} -", layer)


    def summary(self, input_batch_size=torch.Size([1, 3, 224, 224]), verbose=True):
        """
        Display summary of the feature part of the model
        """
        print("input size:", input_batch_size)
        self.model_summary = summary(self.model_features, input_size=input_batch_size)
        if verbose:
            print(self.model_summary)


if __name__ == "__main__":
    # Load a model : AlexNet

    print("No test yet")