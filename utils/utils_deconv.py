import torch

__version__ = "1.1.1"

def make_coherent_before_max_unpool2d(tensor:torch.tensor, pool_indices:torch.tensor):
    """
    Returning a coherent tensor containing the values in the right positions compliant to pool indices, in
    order to get a coherent unmaxpooling.
    """
    tensor_size_length = len(tensor.size())
    pool_indices_size_length = len(pool_indices.size())
    
    assert tensor_size_length in [3, 4], f"tensor size length {tensor_size_length} must be 3 or 4"
    assert pool_indices_size_length in [3, 4], f"pool_indices size length {pool_indices_size_length} must be 3 or 4"
    assert tensor.size() == pool_indices.size(), \
        f"tensor ({tensor.size()}) and pool_indices {pool_indices.size()} sizes must be coherent"
    
    # if no batch
    if tensor_size_length != 4:
        tensor = tensor.unsqueeze(dim=0)
        pool_indices = pool_indices.unsqueeze(dim=0)
    
    tensor_coherent = torch.zeros_like(tensor, device=tensor.device)
    for i_b, (b, p_i_b) in enumerate(zip(tensor, pool_indices)):
        for i_ch, (ch, p_i_ch) in enumerate(zip(b, p_i_b)):
            elts = torch.unique(p_i_ch)
            for e in elts:
                mask = p_i_ch == e
                max_ = (ch * mask).max()
                tensor_coherent[i_b, i_ch] += mask * max_
    
    if tensor_size_length != 4:
        tensor_coherent = tensor_coherent.squeeze(dim=0)
    return tensor_coherent


if __name__ == "__main__":
    # Tests
    tests = [
        { # no batch
            "tensor": torch.tensor([
                [
                    [36,  0, 17, 18],
                    [ 0,  0,  0,  0],
                    [20,  0, 22,  0],
                    [26,  0,  0, 28]
                ]
            ]),
            "pool_indices": torch.tensor([
                [
                    [ 6,  6,  7,  8],
                    [ 6,  6,  7,  8],
                    [10, 11, 12, 13],
                    [16, 16, 17, 18]
                ]
            ]),
            "expected": torch.tensor([
                [
                    [36, 36, 17, 18],
                    [36, 36, 17, 18],
                    [20,  0, 22,  0],
                    [26, 26,  0, 28]
                ]
            ])
        },
        { # batch
            "tensor": torch.tensor([
                [[
                    [36,  0, 17, 18],
                    [ 0,  0,  0,  0],
                    [20,  0, 22,  0],
                    [26,  0,  0, 28]
                ]],
                [[
                    [35,  0, 17,  0],
                    [ 0,  0,  0, 18],
                    [20,  0, 22,  0],
                    [26,  0,  0, 28]
                ]]
            ]),
            "pool_indices": torch.tensor([
                [[
                    [ 6,  6,  7,  8],
                    [ 6,  6,  7,  8],
                    [10, 11, 12, 13],
                    [16, 16, 17, 18]
                ]],
                [[
                    [ 6,  6,  7,  8],
                    [ 6,  6,  7,  8],
                    [10, 11, 12, 13],
                    [16, 16, 17, 18]
                ]]
            ]),
            "expected": torch.tensor([
                [[
                    [36, 36, 17, 18],
                    [36, 36, 17, 18],
                    [20,  0, 22,  0],
                    [26, 26,  0, 28]
                ]],
                [[
                    [35, 35, 17, 18],
                    [35, 35, 17, 18],
                    [20,  0, 22,  0],
                    [26, 26,  0, 28]
                ]]
            ])
        }
    ]

    # tests of make_coherent_before_max_unpool2d
    for test in tests:
        result = make_coherent_before_max_unpool2d(test["tensor"], test["pool_indices"])
        print(result)
        assert torch.equal(result, test["expected"]), "Erreur"
    print("-- Tests of make_coherent_before_max_unpool2d passed", end="\n\n")

    print("== All tests passed =======")
