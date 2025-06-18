import torch
import torch.nn as nn

class L1_loss_weighted(nn.Module):
      def __init__(self,in_shape,device='cpu'):
            super(L1_loss_weighted, self).__init__()
            self.device=device
            self.shape=in_shape

            self.weight=self.get_weight()
            self.weight=self.weight.to(self.device)

      def get_weight(self):
           weights=self.gaussian_blur()
           #weights=self.block_weight()
           return weights 
      
      def gaussian_blur(self):
           #Create weights like gaussian blur filter
           sigma=self.shape[0]/5
           x = torch.arange(0, self.shape[0], 1, dtype=torch.float32)
           y = torch.arange(0, self.shape[1], 1, dtype=torch.float32)
           x, y = torch.meshgrid(x, y, indexing='ij')
           center_x, center_y = self.shape[0] // 2, self.shape[1] // 2
           distances = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
           weights = torch.exp(-(distances**2) / (2 * sigma**2))
           return weights
      
      def block_weight(self):  
          l=15
          factor=0.05
          weights=torch.ones(self.shape)
          weights[:l,:]=factor
          weights[self.shape[0]-l:,l]=factor
          weights[:,:l]=factor
          weights[:,self.shape[1]-l:]=factor
          return weights

      
      
      def forward(self,input,output):
          absolute_diff = torch.abs(input - output)

          weight_diff=self.weight[None,...]*absolute_diff

          loss=torch.mean(weight_diff)

          return loss