from random import randint, seed
import torch

__version__ = "1.0.0"

batch = 3
channel = 3
heigh = 10
width = 10

bc_value = lambda b, c : b * 10 + c
bc_max_value = lambda b, c, cst : bc_value(b, c) * 10 + 7 + cst


def generate_max_and_pos(batch, channel, heigh, width, gen_max=None, one_by=None):
    """
    Retourne tenseur de maximum calculés selon fill_max et leur position aléatoire 3D (batch = 0) ou 4D (batch != 0) par heigh.

    one_by in [None, "tensor", "batch", "channel"]
    """
    pos = []
    max_values = []

    range_b = [randint(0, max(1, batch)-1)] if one_by in ["tensor"] else range(max(1, batch))
    for b in range_b:
        range_c = [randint(0, channel-1)] if one_by in ["tensor", "batch"] else range(channel)
        for c in range_c:
            range_h = [randint(0, heigh-1)] if one_by in ["tensor", "batch", "channel"] else range(heigh)
            for h in range_h:
                pos.append((b, c, h, randint(0, width-1)))
                max_values.append(gen_max(b, c, h) if callable(gen_max) else gen_max)

    pos = torch.tensor(pos)
    max_values = torch.tensor(max_values)

    if batch == 0:
        pos = pos[:, 1:]

    return max_values, pos


def generate_tensor(batch, channel, heigh, width, bc_value, max_values=None, pos=None, dtype=int):
    """
    Retourne un tenseur 3D (batch = 0) ou 4D (batch > 0) remplie selon bc_value, et contenant les max_values à 
    leur position respective indiquée dans pos si fourni.
    """
    assert (max_values!= None and pos != None) or (max_values== None and pos == None), \
        "max_values and pos must be booth not rovided or provided."
    
    t = torch.zeros((max(batch, 1), channel, heigh, width), dtype=dtype)
    for b in range(max(batch, 1)):
        for c in range(channel):
            t[b, c, :, :] = bc_value(b, c) if callable(bc_value) else bc_value

    if b == 0:
        t = t[0]
    
    if max_values != None and pos != None:
        t[*pos.T] = max_values.type(t.dtype)
    
    return t