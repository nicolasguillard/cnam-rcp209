from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision
from .miscellaneous import _overwrite_named_param

__version__ = "1.0.0"

class ConvnetWrapperForDeconvolution():
    def __init__(self, model_to_wrap: nn.Module, features: nn.Module) -> None:
        self.wrapped_model = model_to_wrap
        self.convnet_features = features
        self.set_return_switch_indices(True)
    
    def to(self, device: torch.device) -> None:
        """
        Move all modules in self.convnet_layers to the specified device
        """
        self.wrapped_model.to(device)
        self.features_to_device(device)

    def features_to_device(self, device):
        """
        Move all modules in self.convnet_layers to the specified device
        """
        for m in self.convnet_features:
            m.to(device)

    def forward_through_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the "features" (convnet_layers) part of the model
        """
        x = self.convnet_features(x)
        return x

    def forward_for_deconv(self, 
                x: torch.Tensor,
                idx_layer: int = -1,
                callback_output: Optional[callable] = None,
                return_switch_indices: bool = True,
                verbose: bool = False
                ) -> torch.Tensor|Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """
        Return the forward result AND the collection of (#i, switch indices) for each applyed MaxPool2d in the "features" (convnet_layers) part of the model

        Args:
            x (tensor): input for forward
            idx_layer (int): indice of the module from which to get the ouput, if set
            verbose (bool): if True, print debug messages

        Returns:
            x (tensor): output of the model
            switch_indices (list): list of tuples (i, indices) for each MaxPool2d module in self.convnet_layers
        """
        if idx_layer < 0:
            idx_layer += len(self.convnet_features)
        assert (0 <= idx_layer) and (idx_layer < len(self.convnet_features)), f"i should be in [-{len(self.convnet_features)}; {len(self.convnet_features)}["

        initial_state = self.return_switch_indices
        self.set_return_switch_indices(return_switch_indices)
        switch_indices = []

        #Playing a part of x = self.convnet_layers(x)
        for i, m in enumerate(self.convnet_features):
            if verbose:
                print(f"[{i}] forward ", m)

            if isinstance(m, nn.MaxPool2d) and self.return_switch_indices:
                x, indices = m(x)
                if return_switch_indices:
                    switch_indices.append((i, indices))
            else:
                x = m(x)
            
            if verbose:
                print("\t ouput size:", x.size())
            
            if callback_output is not None:
                callback_output(i, x)

            if i == idx_layer:
                break

        self.set_return_switch_indices(initial_state) # Restore default state
        if return_switch_indices:
            return x, switch_indices
        else:
            return x

    def get_activations(self, 
                x: torch.Tensor,
                coord_activations: Dict[int, List[torch.Tensor]],
                verbose: bool = False
                ) -> Dict[int, List[torch.Tensor]]:
        """
        Return activation values at coord_activations (c, h, w) of each item of batch
        Pool switch indices are not yield

        Args:
            x (torch.Tensor): input for forward
            coord_activations (Dict[int, torch.Tensor]): dictionary of coordinates to get the activations from
                key (int): layer index
                value (List[torch.Tensor]): tensor of coordinates (c, h, w) in order to easily get the activation values from another tensor
            verbose (bool): if True, print debug messages
        """
        # Check coord_activations layer idx
        idx_probed_layers = coord_activations.keys()
        probed_layers_norm_idx_map = {}

        # Normalisation de chaque indice de couche, si négatif, et vérification de la cohérence
        for i, idx in enumerate(idx_probed_layers):
            idx_ = idx + len(self.convnet_features) if idx < 0 else idx
            assert (0 <= idx_) and (idx_ < len(self.convnet_features)), \
                f"idx[{i}] = {idx_} should be in [-{len(self.convnet_features)}; {len(self.convnet_features)}["
            probed_layers_norm_idx_map[idx_] = idx

        ## TODO Check coord_activations chanel, row and col indices w.r.t layer output sizes
        # besoin de fournir les dimensions de l'entrée

        # Creating dict for activations
        activations = {idx_layer:[] for idx_layer in idx_probed_layers}
        sorted_idx_probed_layers = sorted(probed_layers_norm_idx_map)
        idx_layer_max = sorted_idx_probed_layers[-1]
        
        if verbose:
            print("Normalized probed layer indices : ", probed_layers_norm_idx_map)


        def callback_output(idx_layer: int, x: torch.Tensor) -> None:
            if not idx_layer in probed_layers_norm_idx_map:
                if verbose:
                    print(f">(cbk) norm : {idx_layer}")
                return
            
            idx_probed_layer = probed_layers_norm_idx_map[idx_layer]
            if verbose:
                print(f">(cbk) norm : {idx_layer} --> initial {idx_probed_layer}")
            for neuron_coord in coord_activations[idx_probed_layer]:
                # Get the activation value at the specified coordinates
                if verbose:
                    print("\tneuron_coord", neuron_coord)
                chn, row, col = neuron_coord
                # Get the activation value at the specified coordinates
                activations[idx_probed_layer].append(x[:, chn, row, col].to("cpu").detach())
                # activations[idx_layer] = x[*coord_activations[idx_layer].mT]
                #if verbose:
                #    print("activations", activations[idx_probed_layer])

        x = self.forward_for_deconv(
            x,
            idx_layer_max,
            callback_output=callback_output,
            return_switch_indices=False,
            verbose=verbose
            )
        return activations

    def set_return_switch_indices(self, return_indices: bool) -> None:
        """
        Change the return_indices attribut of each Pool2d in self.convnet_layers module group (CNN part)
        """     
        self.return_switch_indices = return_indices
        for m in self.convnet_features:
            if isinstance(m, nn.MaxPool2d):
                m.return_indices = return_indices