from typing import Any, Dict, TypeVar

__version__ = "1.0.0"

# Also visible in PyTorch codes
_pair = lambda v: (v, v)

# From https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
# Necessary in alexnetfordev (evolvment of models.alexnet)
V = TypeVar("V")
def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value

