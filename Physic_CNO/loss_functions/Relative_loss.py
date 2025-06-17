import torch
import torch.nn as nn

class Relative_loss(nn.Module):
      def __init__(self,pad_factor,input_size):
            super(Relative_loss, self).__init__()
            self.pad_factor=pad_factor
            self.original_input_size=input_size-pad_factor

      def forward(self,output_pred,output):
          loss=torch.mean(abs(output_pred[...,:self.original_input_size,:self.original_input_size] \
                              - output)) / torch.mean(abs(output)) * 100
          return loss
      

class Relative_error_training(nn.Module):
      def __init__(self,p,pad_factor,input_size,device):
           super(Relative_error_training,self).__init__()
           self.pad_factor=pad_factor
           self.p=p
           self.original_input_size=input_size-pad_factor
           self.device=device
           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError('p must be 0, 1 or 3 for smooth L1')
            
      def forward(self,output_pred,output):
          loss=self.loss(output_pred[...,:self.original_input_size,:self.original_input_size], output) / self.loss(torch.zeros_like(output).to(self.device), output)
          return loss

    
