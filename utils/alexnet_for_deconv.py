from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision
from .miscellaneous import _ovewrite_named_param

__version__ = "1.0.0"

class AlexNetForDeconv(torchvision.models.AlexNet):
    """
    Evolvement of AlexNet class
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__(num_classes, dropout)
        self.set_return_indices(False)
    

    def features_to_device(self, device):
        for m in self.features:
            m.to(device)
    

    def forward_for_deconv(self, 
                x: torch.Tensor,
                idx_layer: Optional[int] = -1,
                verbose: bool = False
                ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """
        If idx_layer provided, return the forward result AND the collection of (#i, switch indices) for each applyed MaxPool2d in the part "features" of the model

        Args:
            x (tensor): input for forward
            idx_stop (int, optional): indice of the module from which to get the ouput, if set

        Returns:
            
        """
        if idx_layer < 0:
            idx_layer += len(self.features)
        assert (0 <= idx_layer) and (idx_layer < len(self.features)), f"i should be in [-{len(self.features)}; {len(self.features)}["

        self.set_return_indices(True)
        switch_indices = []

        #Playing a part of x = self.features(x)
        for i, m in enumerate(self.features):
            if verbose:
                print(f"[{i}] forward ", m)

            if isinstance(m, nn.MaxPool2d):
                x, indices = m(x)
                switch_indices.append((i, indices))
            else:
                x = m(x)
            
            if verbose:
                print("\t x.size:", x.size())
                        
            if i == idx_layer:
                break

        self.set_return_indices(False) # Restore default state
        return x, switch_indices
    
    def get_max_activations(self,
                            top_n: int = 1,
                            idx_layer_set: int|List[int] = -1 
                            ) -> Dict[int, Tuple[torch.tensor, torch.tensor]]: 
        if type(idx_layer_set) == int:
            idx_layer_set = [idx_layer_set]
        idx_layer_set_ = {}
        for i, idx in enumerate(idx_layer_set):
            idx_ = idx + len(self.features) if idx < 0 else idx
            assert (0 <= idx_) and (idx_ < len(self.features)), \
                f"idx[{i}] = {idx_} should be in [-{len(self.features)}; {len(self.features)}["
            idx_layer_set_[idx_] = idx

        idx_probed_layers = list(probed_layers.keys())
        

    def get_activations(self, 
                x: torch.Tensor,
                coord_activations: Dict[int, torch.Tensor],
                verbose: bool = False
                ) -> Dict[int, torch.Tensor]:
        """
        Return activation values at coord_activations (c, h, w) of each item of batch
        """

        # Check coord_activations layer idx
        probed_layers = {}
        for i, idx in enumerate(coord_activations.keys()):
            idx_ = idx + len(self.features) if idx < 0 else idx
            assert (0 <= idx_) and (idx_ < len(self.features)), \
                f"idx[{i}] = {idx_} should be in [-{len(self.features)}; {len(self.features)}["
            probed_layers[idx_] = idx

        #probed_layers = sorted(probed_layers)
        idx_layer_max = max(probed_layers.keys())

        ## TODO Check coord_activations ch, row, col, wrt output sizes

        # Creating dict for activations
        activations = {}
        idx_probed_layers = list(probed_layers.keys())

        print("idx_probed_layers", idx_probed_layers)

        initial_state = self.return_indices
        self.set_return_indices(False) # Not need of switch indices
        for i, m in enumerate(self.features):
            if verbose:
                print(f"[{i}] forward ", m)

            x = m(x)
            
            if verbose:
                print("\t x.size:", x.size())

            if i in idx_probed_layers:
                idx_probed_layer = probed_layers[i]
                print(">", i, idx_probed_layer)
                activations[idx_probed_layer] = x[*coord_activations[idx_probed_layer].mT]
                        
            if i == idx_layer_max:
                break

        self.set_return_indices(initial_state) # Restore previous state
        return activations


    def set_return_indices(self, return_indices: bool) -> None:
        """
        Change the return_indices attribut of each Pool2d in self.features module group (CNN part)
        """
        
        self.return_indices = return_indices
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
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = AlexNetForDeconv(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model