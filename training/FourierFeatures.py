import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):

    def __init__(self, scale, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale = scale
        self.B = scale * torch.randn((self.mapping_size, 2)).to(device)

    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nx, ny, 2)
        if self.scale != 0:
            x_proj = torch.matmul((2. * np.pi * x), self.B.T)
            inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return inp
        else:
            return x
        
class FourierFeaturesPrecondition(nn.Module):

    def __init__(self, mapping_size, device):
        super().__init__()
        self.mapping_size = mapping_size
        self.scale=torch.arange(1,mapping_size+1, dtype=torch.float32).to(device)
    
    def forward(self, x):
        # x is the set of coordinate and it is passed as a tensor (nx, ny)
        x=x.squeeze(0)
        x_proj = torch.matmul(x.unsqueeze(-1),self.scale.unsqueeze(0))
        inp = 1/np.sqrt(np.pi)*torch.cat([torch.sin(x_proj),1/np.sqrt(2)*torch.ones_like(x).unsqueeze(-1), torch.cos(x_proj)], axis=-1)
        inp = torch.matmul(inp,torch.cat([-1/self.scale**2,torch.tensor([0]),1/self.scale**2],-1))

        return inp
