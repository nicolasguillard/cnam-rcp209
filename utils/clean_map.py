from typing import Optional, Tuple
import torch
from .utils_pytorch import partial_resize_like, get_max_by_dim, get_edge_occurrence_indices, get_first_occurrence_indices, get_last_occurrence_indices

__version__ = "1.5.0"


def cleaning_tensor(
        tensor: torch.tensor, values: torch.tensor = None, pos: torch.tensor = None
        ) -> torch.tensor:
    """
    FR : retourne un tenseur vide sauf pour la ou les valeurs indiquées directement ou par leur position (plus strict).

    tensor (torch.tensor) : shape (batch, chanel, heigh, width)
    value (torch.tensor, optionnal) : value(s) to preserve as [value]. A value by batch or by chanel.
    pos (torch.tensor) : last dimention contains indices of one position (i.e transpose of nonzero(as_tuple=False)) 
        as [(batch, chanel, heigh, width)] or [(chanel, heigh, width)]
    """
    assert (values != None) or (pos != None), f"value or pos must be valued"
    assert (values == None) or (pos == None), f"value and pos must not be valued simultaneously"

    if pos != None:
        mask = torch.zeros_like(tensor)
        mask[*pos.mT] = 1.
        #value_indices = pos.T
        #value = tensor[*value_indices]
    else:
        if len(values) == 1:
            # single value
            values = values.repeat(tensor.size(0))
        # broadcasting value tensor
        if (len(tensor.size()) == 3):
            # without batch
            if len(values.flatten()) == tensor.size(0):
                value_broadcast = values.view(tensor.size(0), 1, 1)
            else:
                value_broadcast = values.view(tensor.size(0), tensor.size(1), 1)
        else:
            # with batch
            if len(values.flatten()) == tensor.size(0):
                value_broadcast = values.view(tensor.size(0), 1, 1 ,1)
            elif len(values.flatten()) == tensor.size(0) * tensor.size(1):
                value_broadcast = values.view(tensor.size(0), tensor.size(1), 1 ,1)
            else:
                value_broadcast = values.view(tensor.size(0), tensor.size(1), tensor.size(2) ,1)

        mask = tensor == value_broadcast

        #print(tensor.size(), value.size(), value_broadcast.size())
        ## indices of occurrences of the value
        #value_indices = (tensor == value_broadcast).nonzero(as_tuple=True)
        #print(value)
        #print(value_indices)

    # !! Ne pas utiliser cette méthode car dans le cas de pos != None, on risque 
    # de fournir plus de valeurs que de pos indiquées !!
    #return tensor * (tensor == value)

    ## keeping this value in the cleaned tensor
    #cleaned = torch.zeros_like(tensor)
    #cleaned[*value_indices] = value
    #return cleaned
    
    cleaned = tensor * mask
    return cleaned


def cleaning_tensor_after_pooling(
        tensor: torch.tensor,
        pool_indices: torch.tensor,
        values: torch.tensor = None,
        pos: torch.tensor = None,
        #max_dim:int = -2,
        keep_only_last_occurrence: bool = False
        ) -> torch.tensor:
    """
    FR : retourne un tenseur vide sauf pour la valeur indiquée, en prenant compte des indices obtenus en appliquant un pooling, afin de préserver la position de la dernière occurrence pertinente par rapport au mécanisme d'unpooling.

    tensor (torch.tensor) : (batch, chanel, heigh, width)
    value (torch.tensor | optionnal) : value(s) to preserve as 
    pos (torch.tensor) : last dimention contains indices of one position (i.e transpose of nonzero(as_tuple=False))
    """
    assert pool_indices != None, "indices from pooling must be provided"
    assert tensor.size() == pool_indices.size(), \
        f"tensor ({tensor.size()}) and pool_indices {pool_indices.size()} sizes must be coherent"
    assert (values != None) or (pos != None), f"values or pos must be valued"
    assert (values == None) or (pos == None), f"values and pos must not be valued simultaneously"

    if pos == None:
        pos = get_first_occurrence_indices(tensor, values)
    
    # Pool indices at pos
    values_pool_indices = torch.full_like(pool_indices, -1) # -1 car ce n'est pas une valeur d'indice de pooling
    values_pool_indices[*pos.T] = pool_indices[*pos.T]

    # dim = -2 search max in a chanel
    MAX_DIM = -2
    values_pool_indices_vector = get_max_by_dim(
        values_pool_indices, dim=MAX_DIM, return_pos=False
        )
    #print(values_pool_indices_vector)

    # Creating the mask
    values_pool_indices_vector_resized = partial_resize_like(values_pool_indices_vector, tensor, MAX_DIM)

    #if len(tensor.size()) == 3 :
    #    values_pool_indices_vector_resized = values_pool_indices_vector.view(tensor.size(0), 1, 1)
    #else:
    #    values_pool_indices_vector_resized = values_pool_indices_vector.view(tensor.size(0), 1, 1, 1)
    
    mask_values_pool_indices = (pool_indices == values_pool_indices_vector_resized)
    cleaned = tensor * mask_values_pool_indices

    if keep_only_last_occurrence:
        #print(values_pool_indices_vector_resized)

        last_occurrence_pool_indices = get_last_occurrence_indices(
            pool_indices, values_pool_indices_vector_resized
            )

        #print(last_occurrence_pool_indices)

        cleaned = torch.zeros_like(tensor)
        cleaned[*last_occurrence_pool_indices.T] = tensor[*last_occurrence_pool_indices.T]

    return cleaned


def cleaning(
        tensor: torch.tensor,
        pos: list = None,
        pool_indices: torch.tensor = None,
        keep_only_last_occurrence: bool = False,
        max_dim: int = -2,
        head_occurrence: bool = True,
        return_pos: bool = False
        ) -> torch.tensor|Tuple[torch.tensor, torch.tensor]:
    """
    Expecting tensor size 3d (c, h, w) or 4d (b, c, h, w).
    FR : retourne un tenseur vide sauf à la position compatible avec la précision indiquée, sinon à la position de la valeur maximum.

    max_dim -> a max by height (-1), chanel (-2) or batch (-3) or tensor (-len(tensor.size()))
    """

    if pos == None:
        # No position provided => get maximums
        if return_pos:
            max_values, max_values_pos = get_max_by_dim(tensor, dim=max_dim, return_pos=True)
        else:
            max_values = get_max_by_dim(tensor, dim=max_dim, return_pos=False)
        max_values = partial_resize_like(max_values, tensor, max_dim)

        # Calcul de positions des premières ou dernières occurrences ?
        if head_occurrence != None:
            pos = get_edge_occurrence_indices(tensor, max_values, head=head_occurrence)
            
    if pool_indices == None:
        if pos != None:
            cleaned = cleaning_tensor(tensor, pos=pos)
        else:
            pos = max_values_pos
            cleaned = cleaning_tensor(tensor, values=max_values)
    else:
        if pos != None:
            cleaned = cleaning_tensor_after_pooling(
                tensor, pool_indices=pool_indices, pos=pos, 
                keep_only_last_occurrence=keep_only_last_occurrence #, max_dim=max_dim
                )
        else:
            pos = max_values_pos
            cleaned = cleaning_tensor_after_pooling(
                tensor, pool_indices=pool_indices, values=max_values, 
                keep_only_last_occurrence=keep_only_last_occurrence, #max_dim=max_dim
                )
            
    if return_pos:
        return cleaned, pos
    else:
        return cleaned
    

def clean_feature_maps(
        feature_maps: torch.Tensor,
        idx_map: int = None,
        pos: Optional[Tuple[int, int]] = None,
        pool_indices: Optional[torch.Tensor] = None,
        return_pos: bool = False,
        keep_only_last_occurrence: bool = False
        ) -> torch.Tensor|Tuple[torch.tensor, torch.tensor]:
    """
    Args:
        - feature_maps (torch.Tensor): assuming size (channels, row, col) or (batch, channels, row, col)
        - idx_map (int) : chanel index. If False, it returns a value by item. If None, return a max value by channel
        - pos ((int, int), optional) : indice of a specific map 
        - pool_indices (torch.Tensor, optional) : if provided, feature maps is generated by a pool2d, indices assuming size (channels, row, col) or (batch, channels, row, col)

    Returns:
        - cleaned feature maps, all zeros tensor but the max value(s)
        - max value
        - Tensor([[chn_max, row_max, col_max]]) : coords of the max
    """
    
    # Si idx_map == False AND pos == None : max of each item of tensor.size(-3), first occurrence of the max 
    # Si idx_map == None AND pos == None : max of each channel
    # Si idx_map == None AND pos != None : same 2d pos in each channel
    # Si idx_map != None AND pos == None : max of the channel idx_map
    # Si idx_map != None AND pos != None : specific position (idx_map, pos(0), pos(1))

    # Si pool_indices vérifier cohérence avec dimensions feature_maps (notamment pour le batch)
    assert len(feature_maps.size()) in [3, 4], "feature_maps size must be 3 (no batchs) or 4 (batchs included)"

    if idx_map == False:
        pos_ = None
        max_dim = -3
    elif idx_map != None and pos == None:
        # Maximum de chaque canal de chaque batch
        max_dim = -2 #-len(feature_maps.size())+2

        max_, max_pos_ = get_max_by_dim(feature_maps, dim=max_dim, return_pos=True)
        # On ne garde que les valeurs des canaux idx_map <=> les autres sont mis à feature_maps.min() - 1
        idx_map_max = torch.full((max_.numel(),), feature_maps.min() - 1)
        idx_map_max[idx_map::feature_maps.size(-3)] = max_.flatten()[idx_map::feature_maps.size(-3)]
        
        idx_map_max = partial_resize_like(idx_map_max, feature_maps, max_dim)
        
        cleaned = cleaning_tensor_after_pooling(
            feature_maps, pool_indices, values=idx_map_max,
            keep_only_last_occurrence=keep_only_last_occurrence #, max_dim=max_dim,
            )
        
        if return_pos:
            return cleaned, max_pos_[idx_map::feature_maps.size(-3)]
        else:
            return cleaned
    else:
        batch_ = len(feature_maps.size()) == 4
        
        if idx_map == None and pos == None:
            pos_ = None    
        elif idx_map != None and pos != None:
            if batch_:
                pos_ = [[b, idx_map, pos[0], pos[1]] for b in range(feature_maps.size(-4))]
            else:
                pos_ = [[idx_map, pos[0], pos[1]]]
        else: #idx_map == None and pos != None
            if batch_:
                pos_ = [[b, ch, pos[0], pos[1]] for b in range(feature_maps.size(-4)) for ch in range(feature_maps.size(-3))]
            else:
                pos_ = [[ch, pos[0], pos[1]] for ch in range(feature_maps.size(-3))]
        
        if pos_ != None:
            pos_ = torch.tensor(pos_)

        max_dim = -2 #-len(feature_maps.size())+2
        #print(pos_)
                
    cleaned = cleaning(
        tensor=feature_maps,
        pos=pos_,
        pool_indices=pool_indices,
        keep_only_last_occurrence=keep_only_last_occurrence,
        max_dim=max_dim,
        return_pos=return_pos
        )
    return cleaned


if __name__ == "__main__":
    # Tests de `cleaning_tensor`
    t = torch.tensor([[[36., 36., 31., 23.],
         [36., 36., 31., 23.],
         [24., 24., 22., 14.],
         [14., 14., 11.,  9.]]])
    value = 31
    pos = torch.tensor([0, 2, 1])
    print(t)

    result = cleaning_tensor(t, values=value)
    print(result)
    expected = torch.tensor([[[ 0.,  0., 31.,  0.],
            [ 0.,  0., 31.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]]])
    assert torch.equal(result, expected), f"Should be {expected}"

    result = cleaning_tensor(t, pos=pos)
    print(result)
    expected = torch.tensor([[[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0., 24.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]]])
    assert torch.equal(result, expected), f"Should be {expected}"
    print("-- Tests of cleaning_tensor passed", end="\n\n")

    # Tests de `cleaning_tensor_after_pooling`
    t = torch.tensor([[
    [36., 36., 31., 23.],
    [36., 36., 31., 23.],
    [24., 24., 22., 14.],
    [14., 14., 11.,  9.]
    ]])
    t_pool_indices = torch.tensor([[
        [ 6,  6,  7,  8],
        [ 6,  6,  7,  8],
        [10, 11, 12, 13],
        [16, 16, 17, 18]
        ]])
    value = 31
    pos1 = torch.tensor([0, 2, 0])
    pos2 = torch.tensor([0, 2, 1])
    print(t)
    print(t_pool_indices)

    result = cleaning_tensor_after_pooling(t, t_pool_indices, values=value)
    print(result)
    expected = torch.tensor([[
        [ 0.,  0., 31.,  0.],
        [ 0.,  0., 31.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]
        ]])
    assert torch.equal(result, expected), f"Should be {expected}"

    result = cleaning_tensor_after_pooling(t, t_pool_indices, values=value, keep_only_last_occurrence=True)
    print(result)
    expected = torch.tensor([[
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0., 31.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]
        ]])
    assert torch.equal(result, expected), f"Should be {expected}"

    result = cleaning_tensor_after_pooling(t, t_pool_indices, pos=pos1)
    print(result)
    expected = torch.tensor([[
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [24.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]
        ]])
    assert torch.equal(result, expected), f"Should be {expected}"

    result = cleaning_tensor_after_pooling(t, t_pool_indices, pos=pos2)
    print(result)
    expected = torch.tensor([[
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0., 24.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]
        ]])
    assert torch.equal(result, expected), f"Should be {expected}"
    print("-- Tests of cleaning_tensor_after_pooling passed", end="\n\n")
    
    print("== All tests passed =======")