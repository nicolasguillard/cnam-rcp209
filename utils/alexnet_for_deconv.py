from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision
from .miscellaneous import _overwrite_named_param

__version__ = "1.1.0"

class AlexNetForDeconv(torchvision.models.AlexNet):
    """
    Evolvement of AlexNet class
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__(num_classes, dropout)
        self.set_return_switch_indices(False)
    

    def features_to_device(self, device):
        """
        Move all modules in self.features to the specified device
        """
        for m in self.features:
            m.to(device)
    
    def forward_through_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the "features" part of the model
        """
        x = self.features(x)
        return x

    def forward_for_deconv(self, 
                x: torch.Tensor,
                idx_layer: int = -1,
                callback_output: Optional[callable] = None,
                return_switch_indices: bool = True,
                verbose: bool = False
                ) -> torch.Tensor|Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """
        Return the forward result AND the collection of (#i, switch indices) for each applyed MaxPool2d in the part "features" of the model

        Args:
            x (tensor): input for forward
            idx_layer (int): indice of the module from which to get the ouput, if set
            verbose (bool): if True, print debug messages

        Returns:
            x (tensor): output of the model
            switch_indices (list): list of tuples (i, indices) for each MaxPool2d module in self.features
        """
        if idx_layer < 0:
            idx_layer += len(self.features)
        assert (0 <= idx_layer) and (idx_layer < len(self.features)), f"i should be in [-{len(self.features)}; {len(self.features)}["

        initial_state = self.return_switch_indices
        self.set_return_switch_indices(return_switch_indices)
        switch_indices = []

        #Playing a part of x = self.features(x)
        for i, m in enumerate(self.features):
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
    
    def get_max_activations(self,
                            top_n: int = 1,
                            idx_layer_set: int|List[int] = -1  
                            ) -> Dict[int, Tuple[torch.tensor, torch.tensor]]:
        """
        Get the top_n activations of each layer in idx_layer_set

        Args:
            top_n (int): number of max activations to get
            idx_layer_set (int|List[int]): layer index or list of layer indices to get the activations from

        Returns:
            activations (Dict[int, Tuple[torch.tensor, torch.tensor]]): dictionary of activations for each layer in idx_layer_set
                key: layer index
                value: tuple of (activations, indices)
        """
        if type(idx_layer_set) == int:
            idx_layer_set = [idx_layer_set]

        pass
        

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
            idx_ = idx + len(self.features) if idx < 0 else idx
            assert (0 <= idx_) and (idx_ < len(self.features)), \
                f"idx[{i}] = {idx_} should be in [-{len(self.features)}; {len(self.features)}["
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
        Change the return_indices attribut of each Pool2d in self.features module group (CNN part)
        """     
        self.return_switch_indices = return_indices
        for m in self.features:
            if isinstance(m, nn.MaxPool2d):
                m.return_indices = return_indices


def alexnet_for_deconv(*,
                  weights: Optional[torchvision.models.AlexNet_Weights] = None,
                  progress: bool = True,
                  **kwargs: Any
                  ) -> AlexNetForDeconv:
    """
    Evolvement of models.alexnet function
    """
    
    weights = torchvision.models.AlexNet_Weights.verify(weights)

    if weights is not None:
        _overwrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = AlexNetForDeconv(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model