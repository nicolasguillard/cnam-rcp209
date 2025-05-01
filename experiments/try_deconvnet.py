# 17/12/2024
# Mais finalement une version fonctionnelle de la déconvolution sera plus pratique

class DeconvNet(nn.Module):
    def __init__(self,
                 from_model: nn.Module,
                 flip_kernels: bool = False,
                 copy_bias: bool = False
                 ) -> None:
        self.maxpool2dcount = 0
        self.deconvnet = self.build_deconvnet(from_model, flip_kernels, copy_bias)
    

    def build_deconvnet(self,
                        from_model: nn.Module,
                        flip_kernels: bool = False,
                        copy_bias: bool = False
                        ) -> nn.Module:
        deconvnet = nn.ModuleList()
        
        for module in reversed(from_model):
            if isinstance(module, nn.MaxPool2d):
                self.maxpool2dcount += 1
                deconvnet.append(nn.MaxUnpool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                ))
            elif isinstance(module, nn.ReLU):
                deconvnet.append(nn.ReLU())
            elif isinstance(module, nn.Conv2d):
                conv = nn.ConvTranspose2d(
                    in_channels=module.out_channels,
                    out_channels=module.in_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    output_padding=1 if module.stride[0] > 1 else 0, # Because stride > 1
                    dilation=module.dilation,
                )
                conv.weight = torch.flip(module.weights, [2, 3]) if flip_kernels else module.weight
                deconvnet.append(conv)

        return deconvnet
    

    def forward(self,
                x: torch.Tensor,
                from_idx_layer: int,
                switch_indices: List[Tuple[int, torch.Tensor]]
                ) -> torch.Tensor:
        """
        Args:
            - from_idx_layer (int) : index of the last forwarded module in the convnet
            - switch_indices (list[tuple[int,torch.Tensor]]) : list of switch indices ordered regarding the way the conv model.
        """
        if from_idx_layer < 0:
            from_idx_layer += len(self.deconvnet)
        assert (0 <= from_idx_layer) and (from_idx_layer < len(self.deconvnet)), f"i should be in [-{len(self.deconvnet)}; {len(self.deconvnet)}["
        from_idx_layer = len(self.deconvnet) - 1 - from_idx_layer

        assert self.maxpool2dcount == len(switch_indices), f"Len(switch_indices) should be {self.maxpool2dcount}"
        
        all_indices = reversed(switch_indices)
        for i, m in enumerate(self.deconvnet):
            if i < from_idx_layer:
                continue
            if isinstance(m, nn.MaxUnpool2d):
                indices = all_indices.pop(0)
                assert indices[0] == i, f"Inconcistency between indices {i}, and {indices[0]} from convent"
                x = m(x, indices[1])
            else:
                x = m(x)
        return x