from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
from PIL import Image

__version__ = "1.2.0"

class UnNormalize(T.Normalize):
    def __init__(self, mean: List[float], std: List[float], *args, **kwargs) -> None:
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def unnormalize(input: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    new_mean = [-m/s for m, s in zip(mean, std)]
    new_std = [1/s for s in std]
    return T.transforms.F.normalize(input, new_mean, new_std)


def to_0_Vmax(image_t: torch.Tensor, val_max: float) -> torch.Tensor:
    return ((image_t - image_t.min()) / (image_t.max() - image_t.min()) * 255.)


def to_0_255(image_t: torch.Tensor) -> torch.Tensor:
    return to_0_Vmax(image_t, 255.)


def to_0_1(image_t: torch.Tensor) -> torch.Tensor:
    return (image_t - image_t.min()) / (image_t.max() - image_t.min())


def display_image_tensor(
        img_tensor: torch.Tensor,
        fn_display: Callable = None,
        resize: Tuple[int, int] = None,
        resample: int = Image.Resampling.NEAREST,
        verbose: bool = True,
        ) -> None:
    """
    Display a image contained in a tensor, assuming dimensions (C, H, W),
    using PIL.
    Dedicated to notebook env : set fn_display=display
    """
    if verbose:
        print("Dimensions :", img_tensor.size(), "\tValeur min :", img_tensor.min().item(), "\tValeur max :", img_tensor.max().item())

    img1 = T.functional.to_pil_image(img_tensor)
    if resize:
        img1 = img1.resize(resize, resample=resample)
    if fn_display:
        fn_display(img1)


def i_suffix_fr(i: int, he: bool = True) -> str:
    f = "er" if he else "ère"
    return f if i==1 else "ème"


def display_images_tensor_grid(
        images: torch.Tensor,
        per_rows: int,
        titles: List[str] = None,
        suptitle: str = "",
        figsize: Tuple[int, int] = (12, 12)
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
    rows = images.size(0) // per_rows + 1
    for r in range(rows):
        for c in range(per_rows):
            i = r * per_rows + c
            if i < rows:
                ax = fig.add_subplot(rows, per_rows, i+1, xticks = [], yticks = [])
                if titles:
                    ax.set_title(titles[i])
                if images[i].size(0) == 1:
                    ax.imshow(images[i].numpy().transpose(1, 2, 0), cmap='gray')
                else:
                    ax.imshow(images[i].numpy().transpose(1, 2, 0))
    if suptitle:
        plt.suptitle(suptitle)
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=1)
    plt.show()

def display_images_list_grid(
        images: List[torch.Tensor],
        per_rows: int,
        titles: List[str] = None,
        suptitle: str = "",
        figsize: Tuple[int, int] = (12, 12)
        ) -> None:
    fig = plt.figure(figsize=figsize, layout='constrained')
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = 1.2
    rows = 1
    for r in range(rows):
        for c in range(per_rows):
            i = r * per_rows + c
            img = images[i]
            ax = fig.add_subplot(rows, per_rows, i+1, xticks = [], yticks = [])
            if titles:
                ax.set_title(titles[i])
            if img.size(0) == 1:
                ax.imshow(img.numpy().transpose(1, 2, 0), cmap='gray')
            else:
                ax.imshow(img.numpy().transpose(1, 2, 0))
    if suptitle:
        plt.suptitle(suptitle)
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=1)
    plt.show()

def show_image_tensor(
        img_tensor: torch.Tensor,
        title: str = "",
        figsize: Tuple[int, int] = (2, 2),
        verbose: bool = True
        ) -> None:
    """
    Display a image contained in a tensor, assuming dimensions (C, H, W),
    using PIL.
    Dedicated to notebook env : set fn_display=display
    """
    if verbose:
        print("Dimensions :", img_tensor.size(), "\tValeur min :", img_tensor.min().item(), "\tValeur max :", img_tensor.max().item())
    titles = [title] if title else None
    display_images_list_grid([img_tensor], 1, titles, figsize=figsize)