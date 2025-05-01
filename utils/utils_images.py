from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T

__version__ = "1.1.1"

class UnNormalize(T.Normalize):
    def __init__(self, mean: List[float], std: List[float], *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def unnormalize(input, mean, std):
    new_mean = [-m/s for m, s in zip(mean,std)]
    new_std = [1/s for s in std]
    return T.transforms.F.normalize(input, new_mean, new_std)

def to_0_255(image_t: torch.Tensor) -> torch.Tensor:
    return (image_t - image_t.min()) / (image_t.max() - image_t.min()) * 255.


def display_image_tensor(img_tensor, verbose=True, fn_display=None):
    """
    Display a image contained in a tensor, assuming dimensions (C, H, W),
    using PIL.
    Dedicated to notebook env : set fn_display=display
    """
    if verbose:
        print("Dimensions", img_tensor.size(), "\nValeur min:", img_tensor.min().item(), "\nValeur max:", img_tensor.max().item())

    img1 = T.functional.to_pil_image(img_tensor)
    if fn_display:
        fn_display(img1)


def i_suffix_fr(i, he=True):
    f = "er" if he else "ère"
    return f if i==1 else "ème"


def display_pictures_grid(
        pictures: torch.Tensor,
        per_rows: int,
        titles: List[str]=None,
        suptitle: str="",
        figsize: Tuple[int, int]=(12, 12)
        ) -> None:
    """ 
    Display a grid of pictures
    
    - Arg(s):
        pictures: torch.Tensor
            Shape (n, C, H, W)
        per_rows: int
        titles: List[str]
            Shape (n) or None
        suptitle: str
        figsize: Tuple[int, int]
    """
    fig = plt.figure(figsize=figsize, layout='constrained')
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = 1.2
    rows = pictures.size(0) // per_rows + 1
    for r in range(rows):
        for c in range(per_rows):
            i = r * per_rows + c
            if i < len(pictures):
                ax = fig.add_subplot(rows, per_rows, i+1, xticks = [], yticks = [])
                if titles:
                    ax.set_title(titles[i])
                if pictures[i].size(0) == 1:
                    ax.imshow(pictures[i].numpy().transpose(1, 2, 0), cmap='gray')
                else:
                    ax.imshow(pictures[i].numpy().transpose(1, 2, 0))
    if suptitle:
        plt.suptitle(suptitle)
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=1)
    plt.show()