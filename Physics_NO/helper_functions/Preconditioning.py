import torch
import sys
sys.path.append('..')
import torch.nn as nn

def Create_P(which_example,s,device,D):
    if which_example=='poisson':
         k_max=s//2
         k_x = torch.cat((torch.arange(start=0, end=k_max+1, step=1,  device=device),
                          torch.arange(start=-k_max+1, end=0, step=1, device=device)), 0).reshape(s, 1).repeat(1, s).reshape(1,s,s)
         k_y = torch.cat((torch.arange(start=0, end=k_max+1, step=1,  device=device),
                          torch.arange(start=-k_max+1, end=0, step=1, device=device)), 0).reshape(1,s).repeat(s, 1).reshape(1,s,s)
         
         #We choose 0 for P00 since of the boundary condition
         #P = (D/2*torch.pi)**2*torch.where((k_x**2+k_y**2) != 0, 1.0 / (k_x**2+k_y**2), 0.0)
         P = 4*D/(torch.pi)**2*torch.where((k_x**2+k_y**2) != 0, 1.0 / (k_x**2+k_y**2), 0.0)
    else:
         raise ValueError('Only poisson equation implemented')
    
    return P

def Precondition_output(output,P):
    k_max=output.shape[-1]//2
    output=torch.fft.irfft2((P[None,...]*torch.fft.fft2(output,dim=[-2,-1]))[...,:k_max+1], dim=[-2, -1])
    return output


class Unnormalize_for_testing:
      def __init__(self,which_example,Normalization_values):
           self.which_example=which_example
           self.Normalization_values=Normalization_values

      def __call__(self,output_pred):
           if self.which_example=='poisson':
                output_pred=(self.Normalization_values["max_model"] - self.Normalization_values["min_model"])*output_pred + self.Normalization_values["min_model"]
           
           elif self.which_example=='helmholtz':
                output_pred = (output_pred*self.Normalization_values["std_model"])+ self.Normalization_values["mean_model"]
           else:
                raise ValueError("Which_example must be Poisson or Helmholtz")
           
           return  output_pred

class Normalize_for_testing:
      def __init__(self,which_example,Normalization_values):
           self.which_example=which_example
           self.Normalization_values=Normalization_values

      def __call__(self,output_pred):
           if self.which_example=='poisson':
                output_pred=(output_pred - self.Normalization_values["min_model"])/(self.Normalization_values["max_model"] - self.Normalization_values["min_model"])
          
           elif self.which_example=='helmholtz':
                output_pred = (output_pred - self.Normalization_values["mean_model"])/self.Normalization_values["std_model"]
           else:
                raise ValueError("Which_example must be Poisson or Helmholtz")
           
           return output_pred



#class Loss_PDE_Precondtioning(nn.Module):
#     def __init__(self,which_example,Normalization_values,p,Boundary_decay,in_size):
#           super(Loss_PDE_Precondtioning, self).__init__()
#           self.which_example=which_example
#           self.Normalization_values=Normalization_values
#           self.p=p
#           assert (p==2)
#           self.Boundary_decay=Boundary_decay
#           self.in_size=in_size
#           self.unnormalize=Unnormalize(self.which_example,self.Normalization_values)
#           
#           if which_example=='poisson':
#              self.D=2
#              self.loss=Poisson_loss(D=self.D,in_size=self.in_size,Boundary_decay=self.Boundary_decay)
#           else:
#               raise ValueError("Which_example must be Poisson or Helmholtz")
#           
#     def forward(self,input,output):
#           input,output=self.unnormalize(input=input,output=output)
#           loss=self.loss(input,output)
#           return loss
#     
#
#class Poisson_loss(nn.Module):
#      '''Calculates the PDE loss for the Poisson equation
#       Input: u  Output of network, Shape = (Batch_size,1,Grid_size,Grid_size)
#              f  Input  of network, Shape = (Batch_size,1,Grid_size,Grid_size)
#              p  Do we use L1 or L2 errors? Default: L1
#              D  Period of Domain
#       Warning: Input f and Output u should not be normalized!'''
#      def __init__(self,D=2,in_size=64,Boundary_decay=1):
#           super(Poisson_loss,self).__init__()
#           self.in_size=in_size
#           self.Boundary_decay=Boundary_decay
#           self.D=D
#           self.loss = torch.nn.MSELoss()
#          
#      def forward(self,input,output): 
#           input=input.squeeze(1)
#           output=output.squeeze(1)
#           
#           loss_pde = self.loss(1/self.D*output,input)
#
#           #Add boundary loss: u=0 on boundary(Domain)
#           boundary_lossx=self.loss(output[...,0,:],torch.zeros_like(output[...,0,:]).to(output[...,0,:].device))
#           boundary_lossy=self.loss(output[...,:,0],torch.zeros_like(output[...,:,0]).to(output[...,:,0].device))
#           boundary_loss=0.5*(boundary_lossx+boundary_lossy)
#           loss=0.5*(loss_pde+boundary_loss*self.Boundary_decay)
#           return loss
#
#
#
#
#
#    
#
#
#         
#    