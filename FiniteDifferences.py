import torch
import torch.nn as nn
import torch.nn.functional as F

class Laplace(nn.Module):
    def __init__(self,s, type = "9-point"):
        super(Laplace,self).__init__()
        self.s=s # define size of the grid
        self.type = type # define type of laplacian
        self.h = 1/self.s # define step size
       
        if self.type == "9-point": # 9-point Patra-Karttunen (default)
            self.laplace = torch.tensor([[1,4,1],
                                         [4,-20,4],
                                         [1,4,1]], dtype=torch.float32).view(1,1,3,3)/(6*self.h**2)
           
        elif self.type == "9-point-OP": # 9-point Oono-Puri
            self.laplace = torch.tensor([[ 1,  2,  1],
                                         [ 2, -12,  2],
                                         [ 1,  2,  1]], dtype=torch.float32).view(1,1,3,3) / (4*self.h**2)
           
        elif self.type == "13-point":
            self.laplace = torch.tensor([[ 0.,   0.,  -1.,   0.,   0.],
                                         [ 0.,   3.,   4.,   3.,   0.],
                                         [-1.,   4., -24.,   4.,  -1.],
                                         [ 0.,   3.,   4.,   3.,   0.],
                                         [ 0.,   0.,  -1.,   0.,   0.]], dtype=torch.float32).view(1, 1, 5, 5) / (6 * self.h**2)
            
        else:
            raise ValueError("Invalid laplacian type, choose from 9-point, 9-point-OP, 13-point")
        
    def forward(self,u):
        return F.conv2d(u,self.laplace,padding=0)
    
    