from random import randint, seed
import torch

import utils.utils_pytorch as utils_pytorch
import utils.clean_map as clean_map
from utils.utils_gen_data import batch, chanel, heigh, width, bc_max_value, bc_value, generate_max_and_pos, generate_tensor
seed(7)

### TEST PARAMS
BATCH_ = 0 # No batch dimension
DIM_ = -3 # One value per t.size(DIM_) regarding BATCH_


### Generate DATA
verbose = True

max_values, max_pos = generate_max_and_pos(BATCH_, chanel, heigh, width, gen_max=bc_max_value, one_by="tensor")
if verbose:
    print("> max_values :", max_values, sep="\n")
    print("> pos :", max_pos, sep="\n")

t = generate_tensor(BATCH_, chanel, heigh, width, bc_value, max_values=max_values, pos=max_pos)
if verbose :
    print("> t :", t.size(), t, sep="\n")

kernel_size=2
stride=1
padding=0
pool_output, pool_indices = torch.nn.functional.max_pool2d(
    t, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True
    )
if verbose:
    print("> output :", pool_output.size(), pool_output, sep="\n")
    print("> pool_indices :", pool_indices.size(), pool_indices, sep="\n")

# récupération du maximum de chaque canal en un vecteur simple
pool_max_values, pool_max_pos = utils_pytorch.get_max_by_dim(pool_output, dim=DIM_, return_pos=True)
if verbose:
    print("> pool_max_values :", pool_max_values.size(), pool_max_values, sep="\n")
    print("> pool_max_pos :", pool_max_pos.size(), pool_max_pos, sep="\n")
    for i, pos in enumerate(pool_max_pos):
        print(f"{i:2d} : {pos}")


### TEST
pos_values = pool_max_pos[[0]]
print("> pos_values", pos_values, sep="\n")
cleaned = clean_map.cleaning_tensor_after_pooling(
    pool_output, pos=pos_values, pool_indices=pool_indices, max_dim=DIM_, keep_only_last_occurrence=False
    )
print("> cleaned", cleaned, sep="\n")
_, pos_max_cleaned = utils_pytorch.get_max_by_dim(cleaned, dim=DIM_, return_pos=True)
expected = pool_max_pos
print(pos_max_cleaned)

assert torch.equal(expected, pos_max_cleaned), "Error"
print("OK !!!")
