from typing import Any, Tuple
from functools import reduce
import torch

__version__ = "1.0.0"


def get_all_occurence_indices(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    """
    return (tensor == value).nonzero(as_tuple=False)


def get_all_max_indices(tensor: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    Return the indices of all occurrences of the maximum value in the tensor.
    """
    max_value = tensor.max().item()
    return max_value, get_all_occurence_indices(tensor, max_value)


def multi_right_unsqueezes(t: torch.tensor, diff_size: int):
    """
    """
    return t.view(list(t.size()) + [1] * diff_size)


def get_max_by_dim(
          tensor: torch.tensor,
          dim: int = -1,
          return_pos: bool = False,
          keepdim: bool = False
          ) -> torch.tensor|Tuple[torch.tensor, torch.tensor]:
    """
        Return a tensor containing max in provided dim, resized regarding dim or not resized,
        and, if required, return max positions in the provided tensor.
    """
    assert -len(tensor.size()) <= dim and dim < 0, f"dim must be in range [-{len(tensor.size())}, -1]"
    
    max_vector = tensor
    for _ in range(-1, dim-1, -1):
        max_vector, _ = torch.max(max_vector, dim=-1, keepdim=False)
    
    #print("D -- ", max_vector.size(), f"dim = {dim}")

    if len(max_vector.size()) == 0:
            max_vector = max_vector[None]
    
    diff = len(tensor.size()) - len(max_vector.size())
    if keepdim:
        max_vector = multi_right_unsqueezes(max_vector, diff)

    #print("D -- ", max_vector.size())

    if return_pos:
        if keepdim:
            max_vector_resized = max_vector
        else :
            diff = len(tensor.size()) - len(max_vector.size())
            max_vector_resized = multi_right_unsqueezes(max_vector, diff)
        
        #print("D -- ", tensor)
        #print("D -- ", max_vector_resized)

        pos = (tensor == max_vector_resized).nonzero(as_tuple=False)
        return max_vector, pos
    else:
        return max_vector
        

def get_edge_occurrence_indices(tensor: torch.tensor, values: torch.tensor, head: bool = True) -> torch.Tensor:
    """
    Return the indices of first occurrence of each values, regarding the shape of values tensor :
    if tensor shape is (b, c, h, w), then if provided values shape is :
        (1),            -> 1 occurrence sur l'ensemble de d1
        (1, 1, 1, 1),   -> 1 occurrence sur l'ensemble de d1
        (b, 1, 1, 1),   -> 1 occurrence par item de d1 (<= card(d1))
        (1, c, 1, 1),   -> 1 occurrence par item de d2 par item de d1 (<= card(d1) * card(d2))
            pattern values repeated on d1
        (b, c, 1, 1),   -> 1 occurrence par item de d2 par item de d1 (<= card(d1)*card(d2))
        (1, 1, h, 1),   -> 1 occurrence par item de d3 par item de d2 par item de d1 (<= card(d1) * card(d2) * card(d3))
            pattern values repeated on d1 and d2
        (b, 1, h, 1),   -> 1 occurrence par item de d3 par item de d2 par item de d1 (<= card(d1) * card(d2) * card(d3))
            pattern values repeated on d2 for each d1
        (b, c, h, 1)    -> 1 occurrence par item de d3 par item de d2 par item de d1 (<= card(d1) * card(d2) * card(d3))
    """
    idx_edge = 0 if head else -1

    # détection des valeurs recherchées dans les dimensions corresondantes selon la forme de values
    pos = (tensor == values).nonzero()

    # squeezing by right as long as possible
    v_size = list(values.size())
    while len(v_size) > 1 and v_size[-1] == 1:
        v_size.pop()

    # finding first occurrence positions
    len_v_size = len(v_size)
    t_size = tensor.size()[:len_v_size]
    indices = torch.zeros((len_v_size,), dtype=int)
    first_occ_pos = []
    # for each possible indices regarding dimensions of tensor
    while indices[0] < t_size[0]:
        #print("indices", indices)
        # search pos starting with indices
        eligible_pos = (pos[:, :len_v_size] == indices).all(dim=1)
        #pos_eligible_pos = eligible_pos.nonzero(as_tuple=False)
        #print(eligible_pos)
        #print(eligible_pos.any())
        if eligible_pos.any():
            first_occ_pos.append(pos[eligible_pos][idx_edge].tolist())
        # get first occurrence position
        #print(pos_eligible_pos)
        #if (len(pos_eligible_pos)):
        #    first_occ_pos.append(pos[*pos_eligible_pos[0]].tolist())
        # increment indices
        d = -1
        while d < 0:
            indices[d] += 1
            if indices[d] == t_size[d] and d > -len_v_size:
                indices[d] = 0
                d -= 1
            else:
                d = 0
    #print(first_occ_pos)
    if len_v_size == 1 and v_size[0] == 1:
        return torch.tensor(first_occ_pos[:1])
    else:
        return torch.tensor(first_occ_pos)


def get_first_occurrence_indices(tensor: torch.tensor, values: torch.tensor) -> torch.Tensor:
    return get_edge_occurrence_indices(tensor, values, True)


def get_last_occurrence_indices(tensor: torch.tensor, values: torch.tensor) -> torch.Tensor:
    return get_edge_occurrence_indices(tensor, values, False)


def partial_resize_like(tensor: torch.tensor, like_tensor: torch.tensor, max_dim: int):
    len_tensor_like_size = len(like_tensor.size())
    assert max_dim >= -len_tensor_like_size and max_dim <= len_tensor_like_size, \
        f"max_dim {max_dim} mus be in range [{-len_tensor_like_size}; {-len_tensor_like_size}]"
    if max_dim <= 0:
        max_dim = len_tensor_like_size + max_dim
    resize = [1] * len_tensor_like_size
    for d in range(0, max_dim):
        resize[d] = like_tensor.size(d)
    return tensor.view(resize)


if __name__ == "__main__":
    # Tests
    size = [b, c, h, w] = [2, 3, 2, 5]

    n = reduce(lambda x,y : x*y, size[2:])
    t_4d_r = torch.arange(0, n).view(1, 1, h, w).repeat(2, 3, 1, 1)
    t_4d_r[:, :, 1, 3] = 1
    print("> t_4d_r")
    print(t_4d_r.size(), t_4d_r, sep="\n")

    t_3d_r = t_4d_r[0]
    print("> t_3d_r")
    print(t_3d_r.size(), t_3d_r, sep="\n")

    print("> t_4d_d")
    n = reduce(lambda x,y : x*y, size)
    t_4d_d = torch.arange(0, n).view(size)
    print(t_4d_d.size(), t_4d_d, sep="\n")

    t_3d_d = t_4d_d[0]
    print("> t_3d_d")
    print(t_3d_d.size(), t_3d_d, sep="\n")

    # Tests of get_all_max_indices
    tests = [
        { # 1
            "tensor": t_3d_r,
            "values": torch.tensor([1]),
            "expected": torch.tensor([1])
        },

        { #
            "tensor": t_3d_r,
            "values": torch.tensor([1]).view(1, 1, 1).repeat(c, 1, 1),
            "expected": torch.tensor([1]).repeat(c)
        },
        { #
            "tensor": t_3d_d,
            "values": torch.tensor([1]).view(1, 1, 1).repeat(c, 1, 1),
            "expected": torch.tensor([1])
        },
        { #
            "tensor": t_3d_r,
            "values": torch.tensor([1]).view(1, 1, 1).repeat(c, h, 1),
            "expected": torch.tensor([1]).repeat(c*h)
        },
        { #
            "tensor": t_3d_d,
            "values": torch.tensor([1]).view(1, 1, 1).repeat(c, h, 1),
            "expected": torch.tensor([1])
        },

        { #
            "tensor": t_3d_r,
            "values": torch.tensor([1, 7, 9]).view(c, 1, 1),
            "expected": torch.tensor([1, 7, 9])
        },
        { #
            "tensor": t_3d_d,
            "values": torch.tensor([7, 17, 28]).view(c, 1, 1),
            "expected": torch.tensor([7, 17, 28])
        },

        { #
            "tensor": t_3d_r,
            "values": torch.tensor([1, 6]).view(1, h, 1).repeat(c, 1, 1),
            "expected": torch.tensor([1, 6]).repeat(3)
        },
        { #
            "tensor": t_3d_d,
            "values": torch.tensor([2, 7, 12, 16, 23, 28]).view(c, h, 1),
            "expected": torch.tensor([2, 7, 12, 16, 23, 28])
        },

        { # 1
            "tensor": t_4d_r,
            "values": torch.tensor([1]),
            "expected": torch.tensor([1])
        },

        { #
            "tensor": t_4d_r,
            "values": torch.tensor([1]).view(1, 1, 1, 1).repeat(b, 1, 1, 1),
            "expected": torch.tensor([1]).repeat(b)
        },
        { #
            "tensor": t_4d_d,
            "values": torch.tensor([1]).view(1, 1, 1, 1).repeat(b, 1, 1, 1),
            "expected": torch.tensor([1])
        },
        { #
            "tensor": t_4d_r,
            "values": torch.tensor([1]).view(1, 1, 1, 1).repeat(b, c, 1, 1),
            "expected": torch.tensor([1]).repeat(b*c)
        },
        { #
            "tensor": t_4d_d,
            "values": torch.tensor([1]).view(1, 1, 1, 1).repeat(b, c, 1, 1),
            "expected": torch.tensor([1])
        },
        { #
            "tensor": t_4d_r,
            "values": torch.tensor([1]).view(1, 1, 1, 1).repeat(b, c, h, 1),
            "expected": torch.tensor([1]).repeat(b*c*h)
        },
        { #
            "tensor": t_4d_d,
            "values": torch.tensor([1]).view(1, 1, 1, 1).repeat(b, c, h, 1),
            "expected": torch.tensor([1])
        },

        { #
            "tensor": t_4d_r,
            "values": torch.tensor([1, 7]).view(b, 1, 1, 1),
            "expected": torch.tensor([1, 7])
        },
        { #
            "tensor": t_4d_d,
            "values": torch.tensor([7, 37]).view(b, 1, 1, 1),
            "expected": torch.tensor([7, 37])
        },

        { #
            "tensor": t_4d_r,
            "values": torch.tensor([1, 6, 9]).view(1, c, 1, 1).repeat(b, 1, 1, 1),
            "expected": torch.tensor([1, 6, 9, 1, 6, 9])
        },
        { #
            "tensor": t_4d_d,
            "values": torch.tensor([7, 12, 23, 37, 46, 55]).view(b, c, 1, 1),
            "expected": torch.tensor([7, 12, 23, 37, 46, 55])
        },
        { #
            "tensor": t_4d_d,
            "values": torch.tensor([2, 7, 12, 16, 23, 28, 31, 37, 40, 46, 54, 55]).view(b, c, h, 1),
            "expected": torch.tensor([2, 7, 12, 16, 23, 28, 31, 37, 40, 46, 54, 55])
        },
    ]

    for i, test in enumerate(tests, start=1):
        #print(f"=== Test #{i}")
        #print("> tensor :", test["tensor"].size(), sep="\n")
        result = get_first_occurrence_indices(test["tensor"], test["values"])
        #print("> result :", result, sep="\n")
        check = test["tensor"][*result.T]
        #print("> check :", check, sep="\n")
        assert torch.equal(test["expected"], check), f"Test #{i} : Error {test['expected']}"
        #print()

    print("-- Tests of get_all_max_indices passed", end="\n\n")

    print("== All tests passed =======")