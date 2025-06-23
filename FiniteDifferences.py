import torch
import torch.nn as nn
import torch.nn.functional as F

class Laplace(nn.Module):
    def __init__(self,s, D=1.0, type = "9-point"):
        super(Laplace,self).__init__()
        self.s=s # define size of the grid
        self.D = D # define domain size
        self.h = self.D/self.s # define step size
        self.type = type # define type of laplacian
       
        if self.type == "9-point": # 9-point Patra-Karttunen (default)
            self.laplace = torch.tensor([[1,4,1],
                                         [4,-20,4],
                                         [1,4,1]], dtype=torch.float32).view(1,1,3,3)/(6*self.h**2)
            self.cut_size = 1
        elif self.type == "9-point-OP": # 9-point Oono-Puri
            self.laplace = torch.tensor([[ 1,  2,  1],
                                         [ 2, -12,  2],
                                         [ 1,  2,  1]], dtype=torch.float32).view(1,1,3,3) / (4*self.h**2)
            self.cut_size = 1
        elif self.type == "13-point":
            self.laplace = torch.tensor([[ 0.,   0.,  -1.,   0.,   0.],
                                         [ 0.,   3.,   4.,   3.,   0.],
                                         [-1.,   4., -24.,   4.,  -1.],
                                         [ 0.,   3.,   4.,   3.,   0.],
                                         [ 0.,   0.,  -1.,   0.,   0.]], dtype=torch.float32).view(1, 1, 5, 5) / (6 * self.h**2)
            self.cut_size = 2
        else:
            raise ValueError("Invalid laplacian type, choose from 9-point, 9-point-OP, 13-point")
        
    def forward(self,u):
        # Handle both cases: with and without channel dimension
        # u can be [batch, height, width] or [batch, channels, height, width]
        if u.dim() == 3:
            # Add channel dimension for conv2d: [batch, height, width] -> [batch, 1, height, width]
            u_input = u.unsqueeze(1)
            squeeze_output = True
        elif u.dim() == 4:
            # Already has channel dimension
            u_input = u
            squeeze_output = False
        else:
            raise ValueError(f"Expected 3D or 4D input tensor, got {u.dim()}D")
        
        # Move laplace kernel to same device as input
        if self.laplace.device != u_input.device:
            self.laplace = self.laplace.to(u_input.device)
        
        # Apply convolution
        laplace_result = F.conv2d(u_input, self.laplace, padding=0)
        
        # Remove channel dimension if input didn't have it
        if squeeze_output:
            laplace_result = laplace_result.squeeze(1)
            
        return laplace_result, self.cut_size
    
    