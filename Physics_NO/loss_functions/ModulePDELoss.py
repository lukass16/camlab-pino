import torch
import torch.nn as nn
import sys
sys.path.append('..')
from Physics_NO.helper_functions.Preconditioning import Precondition_output, Create_P

from FiniteDifferences import Laplace as FDLaplace

class Loss_PDE(nn.Module):
    def __init__(self,which_example,Normalization_values,p,pad_factor,in_size,preconditioning=False,device='gpu', force_fourier=False, force_fd=False):
       super(Loss_PDE, self).__init__()
       self.which_example=which_example # sets which dataset is being used
       self.Normalization_values=Normalization_values
       self.p=p 
       self.pad_factor=pad_factor
       self.in_size=in_size # input grid size for the model
       self.preconditioning=preconditioning
       self.unnormalize=Unnormalize(self.which_example,self.Normalization_values) # define our unnormalization function
       self.force_fourier=force_fourier
       self.force_fd=force_fd
       
       if self.force_fourier and self.force_fd:
          raise ValueError("force_fourier and force_fd cannot be True at the same time")
       
       if which_example=='poisson':
          self.D=1 # ! changed to 1 since we are using a [0,1] domain
          self.loss=Poisson_loss_FD(p=self.p,D=self.D,in_size=self.in_size,pad_factor=self.pad_factor)
             
       elif which_example=='helmholtz':
          self.D=1
          self.loss=Helmholtz_loss_FD(p=self.p,D=self.D,in_size=self.in_size,pad_factor=self.pad_factor)
          
          # deal with force_fourier and force_fd #! this is for debugging purposes
          if self.force_fourier:
             self.loss=Helmholtz_loss(p=self.p,D=self.D,in_size=self.in_size,pad_factor=self.pad_factor)
          elif self.force_fd:
             self.loss=Helmholtz_loss_FD(p=self.p,D=self.D,in_size=self.in_size,pad_factor=self.pad_factor)
       else:
           raise ValueError("Which_example must be Poisson or Helmholtz")
       
       if self.preconditioning:
           self.P=Create_P(which_example=which_example,s=in_size,device=device,D=self.D)
       
    def forward(self,input,output):
        input,output=self.unnormalize(input=input,output=output) # for pde loss need to use unnormalized data

        # if using preconditioning
        if self.preconditioning:
           output=Precondition_output(output,self.P)
        
        # get back pde and boundary loss
        loss_pde, loss_boundary=self.loss(input=input,output=output) # calculate the loss
        return loss_pde, loss_boundary
        
class Loss_OP(nn.Module):
      def __init__(self,p,in_size,pad_factor):
          super(Loss_OP, self).__init__()
          self.p=p
          self.original_input_size=in_size-pad_factor
          if p == 1:
             self.loss = torch.nn.L1Loss()
          elif p == 2:
             self.loss = torch.nn.MSELoss()
          elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
          else:
             raise ValueError("p must be 1, 2 or 3 for smooth L1")
          
      def forward(self,output_train,output_fix):
           loss = self.loss(output_train[...,:self.original_input_size,:self.original_input_size],\
                             output_fix[...,:self.original_input_size,:self.original_input_size])
           return loss
      
class Unnormalize:
      def __init__(self,which_example,Normalization_values):
           self.which_example=which_example
           self.Normalization_values=Normalization_values

      def __call__(self,input,output):
           if self.which_example=='poisson':
                output=(self.Normalization_values["max_model"] - self.Normalization_values["min_model"])*output + self.Normalization_values["min_model"]
                input =(self.Normalization_values["max_data"]  - self.Normalization_values["min_data"]) *input  +  self.Normalization_values["min_data"]
           
           elif self.which_example=='helmholtz':
                input=input+self.Normalization_values["mean_data"]
                output = (output*self.Normalization_values["std_model"])+ self.Normalization_values["mean_model"]
           
           else:
                raise ValueError("Which_example must be Poisson or Helmholtz")
           
           return input, output
                

class Poisson_loss_FD(nn.Module):
      '''Calculates the PDE loss for the Poisson equation
       Input: u  Output of network, Shape = (Batch_size,1,Grid_size,Grid_size)
              f  Input  of network, Shape = (Batch_size,1,Grid_size,Grid_size)
              p  Do we use L1 or L2 errors? Default: L1
              D  Period of Domain
       Warning: Input f and Output u should not be normalized!'''
      def __init__(self,p=1,D=1.0,in_size=64,pad_factor=0):
           super(Poisson_loss_FD,self).__init__()
           self.p=p
           self.in_size=in_size
           self.pad_factor=pad_factor
           self.original_input_size=self.in_size-self.pad_factor
           self.D=D+self.pad_factor/(self.in_size-self.pad_factor)*D
           self.Laplace=FDLaplace(s=in_size,D=self.D) #! changed to FDLaplace - testing Finite Diff Laplacian

           # choose what type of loss we want to use
           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError("p must be 1 for L1, 2 for L2 or 3 for smooth L1")
          
      def forward(self,input,output): 
           input=input.squeeze(1)
           output=output.squeeze(1)
           Laplace_u, cut_size =self.Laplace(output) # get laplace of the output
           
           loss_pde = self.loss(-Laplace_u,input[...,cut_size:-cut_size,cut_size:-cut_size]) #note: we slice the last two spatial dimensions removing boundary

           #Add boundary loss: u=0 on boundary(Domain)
           boundary_lossx=self.loss(output[...,0,:],torch.zeros_like(output[...,0,:]).to(output[...,0,:].device))
           boundary_lossy=self.loss(output[...,:,0],torch.zeros_like(output[...,:,0]).to(output[...,:,0].device))
           
           boundary_lossx_D=self.loss(output[...,-1,:],torch.zeros_like(output[...,-1,:]).to(output[...,-1,:].device)) #! added other boundary to also be 0
           boundary_lossy_D=self.loss(output[...,:,-1],torch.zeros_like(output[...,:,-1]).to(output[...,:,-1].device)) #! added other boundary to also be 0
           
           
           boundary_loss=0.25*(boundary_lossx+boundary_lossy+boundary_lossx_D+boundary_lossy_D) # take average of the boundary losses for x and y directions
           return loss_pde,boundary_loss
                          
#todo: readd the Poisson_loss with fourier laplacian for comparison

class Laplace(nn.Module):
    '''Calculates Laplace
    Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
    Ouput: Delta u '''
    def __init__(self,s,D):
        super(Laplace,self).__init__()
        self.s=s
        self.D=D
    def forward(self,u):
         #Calculate derivative using the fact FT(f')=2*pi*i*k*FT(f)
         u_hat=torch.fft.fft2(u,dim=[-2,-1])
         
         assert (u.device==u_hat.device) #Need to be same device, can only be checked on GPU
     
         k_max=self.s//2
         
         k_x = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                          torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(self.s, 1).repeat(1, self.s).reshape(1,self.s,self.s)
         k_y = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                          torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(1, self.s).repeat(self.s, 1).reshape(1,self.s,self.s)
     
         #Calculate Laplace of u
         Laplace_u_hat =-4*(torch.pi/self.D)**2*(k_x**2+k_y**2)*u_hat

         Laplace_u=torch.fft.irfft2(Laplace_u_hat[...,:k_max + 1], dim=[-2, -1])
    
         return Laplace_u
     
class Helmholtz_hybrid_loss_FD(nn.Module):
    def __init__(self,omega=5*torch.pi/2,p=1,D=1,pad_factor=0,in_size=128):
           super(Helmholtz_hybrid_loss_FD,self).__init__()
           self.omega=omega
           self.p=p
           self.pad_factor=pad_factor
           self.in_size=in_size
           self.original_input_size=self.in_size-self.pad_factor
           self.D=D+self.pad_factor/(self.in_size-self.pad_factor)*D #D for extended domain
           self.Laplace=FDLaplace(s=self.in_size,D=self.D) #! changed to FDLaplace - testing Finite Diff Laplacian

           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError("p must be 1, 2 or 3 for smooth L1")

    def forward(self,input,output,label): # note: these input and output are already unnormalized
        assert output.dim() == 4 and output.shape[1] == 1, f"output must be of shape (B, 1, H, W) but got {output.shape}" #! debug
        
        #Add boundary loss
        a=input[:,0,:,:]
        output = output.squeeze(1)
        label = label.squeeze(1)
            
        boundary=input[:,1,0,0].unsqueeze(-1).unsqueeze(-1) # (B, 1, 1) - all the boundaries for the batch
        
        boundary_line = boundary.expand(-1, -1, self.original_input_size).squeeze(1) # shape (B, grid_size) - constant boundary line for each batch

        # match the boundary to the boundary conditions
        boundary_lossx_0=self.loss(output[:,0,:],  boundary_line)
        boundary_lossy_0=self.loss(output[:,:,0],  boundary_line)
    
        boundary_lossx_D=self.loss(output[:,-1,:], boundary_line)
        boundary_lossy_D=self.loss(output[:,:,-1], boundary_line)
        boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
    
        # calculate the laplace of the output and label
        Laplace_u_output, cut_size =self.Laplace(output)	
        Laplace_u_label, cut_size =self.Laplace(label)

        # calculate the pde loss (Laplace_u = -omega^2 * a^2 * output)
        loss_pde_1=self.loss(Laplace_u_output,\
                        -self.omega**2*a[...,cut_size:-cut_size,cut_size:-cut_size]**2*label[...,cut_size:-cut_size,cut_size:-cut_size])
        
        # loss_pde_2=self.loss(Laplace_u_output,\
        #                 -self.omega**2*a[...,cut_size:-cut_size,cut_size:-cut_size]**2*label[...,cut_size:-cut_size,cut_size:-cut_size])
        
        loss_pde=loss_pde_1#+loss_pde_2
        
        #! DEBUG
        #print(f"loss_pde_1: {loss_pde_1.item()}, loss_pde_2: {loss_pde_2.item()}")
        
    
        return loss_pde, boundary_loss

    
class Helmholtz_loss_FD(nn.Module):
    def __init__(self,omega=5*torch.pi/2,p=1,D=1,pad_factor=0,in_size=128):
           super(Helmholtz_loss_FD,self).__init__()
           self.omega=omega
           self.p=p
           self.pad_factor=pad_factor
           self.in_size=in_size
           self.original_input_size=self.in_size-self.pad_factor
           self.D=D+self.pad_factor/(self.in_size-self.pad_factor)*D #D for extended domain
           self.Laplace=FDLaplace(s=self.in_size,D=self.D) #! changed to FDLaplace - testing Finite Diff Laplacian

           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError("p must be 1, 2 or 3 for smooth L1")

    def forward(self,input,output): # note: these input and output are already unnormalized
        assert output.dim() == 4 and output.shape[1] == 1, f"output must be of shape (B, 1, H, W) but got {output.shape}" #! debug
        
        #Add boundary loss
        a=input[:,0,:,:]
        output = output.squeeze(1)
            
        boundary=input[:,1,0,0].unsqueeze(-1).unsqueeze(-1) # (B, 1, 1) - all the boundaries for the batch
        
        boundary_line = boundary.expand(-1, -1, self.original_input_size).squeeze(1) # shape (B, grid_size) - constant boundary line for each batch

        # match the boundary to the boundary conditions
        boundary_lossx_0=self.loss(output[:,0,:],  boundary_line)
        boundary_lossy_0=self.loss(output[:,:,0],  boundary_line)
    
        boundary_lossx_D=self.loss(output[:,-1,:], boundary_line)
        boundary_lossy_D=self.loss(output[:,:,-1], boundary_line)
        boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
    
        # calculate the laplace of the output
        Laplace_u, cut_size =self.Laplace(output)	

        # calculate the pde loss (Laplace_u = -omega^2 * a^2 * output)
        loss_pde=self.loss(Laplace_u,\
                        -self.omega**2*a[...,cut_size:-cut_size,cut_size:-cut_size]**2*output[...,cut_size:-cut_size,cut_size:-cut_size])
    
        return loss_pde, boundary_loss
       
       
class Helmholtz_loss(nn.Module):
      def __init__(self,omega=5*torch.pi/2,p=1,D=1,pad_factor=0,in_size=128):
           super(Helmholtz_loss,self).__init__()
           self.omega=omega
           self.p=p
           self.pad_factor=pad_factor
           self.in_size=in_size
           self.original_input_size=self.in_size-self.pad_factor
           self.D=D+self.pad_factor/(self.in_size-self.pad_factor)*D #D for extended domain
           self.Laplace=Laplace(s=self.in_size,D=self.D)

           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError("p must be 1, 2 or 3 for smooth L1")
           

      def forward(self,input,output):
           #Add boundary loss
           a=input[:,0,:,:]
           if output.dim() == 4 and output.shape[1] > 1:
               output = output[:, 0] # Take first channel, shape becomes (B, H, W)
               print(f"output shape: {output.shape}") #! debug
           else:
               output = output.squeeze(1)
           boundary=input[:,1,0,0].unsqueeze(-1).unsqueeze(-1)
           
           # match the boundary to the coundary conditions
           boundary_lossx_0=self.loss(output[...,0,:self.original_input_size],  boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_lossy_0=self.loss(output[...,:self.original_input_size,0],  boundary.expand(-1, -1, self.original_input_size).squeeze(1))
       
           boundary_lossx_D=self.loss(output[...,-1-self.pad_factor,:self.original_input_size], boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_lossy_D=self.loss(output[...,:self.original_input_size,-1-self.pad_factor], boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
       
           # calculate the laplace of the output
           Laplace_u=self.Laplace(output)	

           # calculate the pde loss (Laplace_u = -omega^2 * a^2 * output)
           loss_pde=self.loss(Laplace_u[...,:self.original_input_size,:self.original_input_size],\
                            -self.omega**2*a[...,:self.original_input_size,:self.original_input_size]**2*output[...,:self.original_input_size,:self.original_input_size])
       
           return loss_pde, boundary_loss
      
#-----------------------------------------------Additonal------------------------------------------------
class Inverse_Laplace(nn.Module):
    '''Calculates Inverse Laplace
    Input: torch.tensor of Shape = (Batch_size,Grid_size,Grid_size)
    Ouput: 1/Delta u '''
    def __init__(self,s,D):
        super(Inverse_Laplace,self).__init__()
        self.s=s
        self.D=D
    def forward(self,u):
         #Calculate derivative using the fact FT(f')=2*pi*i*k*FT(f)
         u_hat=torch.fft.fft2(u,dim=[-2,-1])
         
         assert (u.device==u_hat.device) #Need to be same device, can only be checked on GPU
     
         k_max=self.s//2
         
         k_x = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                          torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(self.s, 1).repeat(1, self.s).reshape(1,self.s,self.s)
         k_y = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                          torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(1, self.s).repeat(self.s, 1).reshape(1,self.s,self.s)
     
         #Calculate Laplace of u
         Laplace_u_hat =-4*(torch.pi/self.D)**2*(k_x**2+k_y**2)
         Laplace_u_hat[...,0,0]=1

         Inverse_Laplace_u_hat=1/Laplace_u_hat*u_hat
         
         Inverse_Laplace_u=torch.fft.irfft2(Inverse_Laplace_u_hat[...,:k_max + 1], dim=[-2, -1])
    
         return Inverse_Laplace_u


class Helmholtz_loss_with_inverse(nn.Module):
      def __init__(self,omega=5*torch.pi/2,p=1,D=1,Boundary_decay=1,pad_factor=0,in_size=128):
           super(Helmholtz_loss_with_inverse,self).__init__()
           self.omega=omega
           self.p=p
           self.Boundary_decay=Boundary_decay
           self.pad_factor=pad_factor
           self.in_size=in_size
           self.original_input_size=self.in_size-self.pad_factor
           self.D=D+self.pad_factor/(self.in_size-self.pad_factor)*D #D for extended domain
           self.Laplace=Inverse_Laplace(s=self.in_size,D=self.D)

           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError("p must be 1, 2 or 3 for smooth L1")
           

      def forward(self,input,output):
           #Add boundary loss
           a=input[:,0,:,:]
           output=output.squeeze(1)
           boundary=input[:,1,0,0].unsqueeze(-1).unsqueeze(-1)
           assert (self.pad_factor==0)
           boundary_lossx_0=self.loss(output[...,0,:self.original_input_size],  boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_lossy_0=self.loss(output[...,:self.original_input_size,0],  boundary.expand(-1, -1, self.original_input_size).squeeze(1))
       
           boundary_lossx_D=self.loss(output[...,-1-self.pad_factor,:self.original_input_size], boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_lossy_D=self.loss(output[...,:self.original_input_size,-1-self.pad_factor], boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
       
           Laplace_asquared=self.Laplace(-self.omega**2*a[...,:self.original_input_size,:self.original_input_size]**2*output[...,:self.original_input_size,:self.original_input_size])
        
           loss_pde=self.loss(output[...,:self.original_input_size,:self.original_input_size],\
                              Laplace_asquared)


           loss=loss_pde+boundary_loss*self.Boundary_decay
       
           return loss
      
class Helmholtz_loss_op(nn.Module):
      def __init__(self,omega=5*torch.pi/2,p=1,D=1,Boundary_decay=1,pad_factor=0,in_size=128):
           super(Helmholtz_loss_op,self).__init__()
           self.omega=omega
           self.p=p
           self.Boundary_decay=Boundary_decay
           self.pad_factor=pad_factor
           self.in_size=in_size
           self.original_input_size=self.in_size-self.pad_factor
           self.D=D+self.pad_factor/(self.in_size-self.pad_factor)*D #D for extended domain
           self.Laplace=Laplace(s=self.in_size,D=self.D)

           if p == 1:
               self.loss = torch.nn.L1Loss()
           elif p == 2:
               self.loss = torch.nn.MSELoss()
           elif p ==3:
               self.loss = torch.nn.SmoothL1Loss()
           else:
               raise ValueError("p must be 1, 2 or 3 for smooth L1")
           

      def forward(self,input,output,output_fix):
           #Add boundary loss
           a=input[:,0,:,:]
           output=output.squeeze(1)
           output_fix=output_fix.squeeze(1)
           boundary=input[:,1,0,0].unsqueeze(-1).unsqueeze(-1)
           
           boundary_lossx_0=self.loss(output[...,0,:self.original_input_size],  boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_lossy_0=self.loss(output[...,:self.original_input_size,0],  boundary.expand(-1, -1, self.original_input_size).squeeze(1))
       
           boundary_lossx_D=self.loss(output[...,-1-self.pad_factor,:self.original_input_size], boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_lossy_D=self.loss(output[...,:self.original_input_size,-1-self.pad_factor], boundary.expand(-1, -1, self.original_input_size).squeeze(1))
           boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
       
           Laplace_u=self.Laplace(output)
           assert (Laplace_u[...,:self.original_input_size,:self.original_input_size].size()==a[...,:self.original_input_size,:self.original_input_size].size())
           loss_pde=self.loss(Laplace_u[...,:self.original_input_size,:self.original_input_size],\
                            -self.omega**2*a[...,:self.original_input_size,:self.original_input_size]**2*output_fix[...,:self.original_input_size,:self.original_input_size])


           loss=loss_pde+boundary_loss*self.Boundary_decay
       
           return loss