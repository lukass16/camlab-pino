import torch

def transform(input):
        s=input.shape[-1]
        input=torch.nn.functional.interpolate(input, size=(s+1,s+1),mode='bilinear')
        input=input[...,:-1,:-1]
        return input

def loss_pde(u,f,which_example,Normalization_values,p=1,Boundary_decay=None,pad_factor=0):
    if which_example=='poisson':
       u=(Normalization_values["max_model"] - Normalization_values["min_model"])*u + Normalization_values["min_model"]
       f=(Normalization_values["max_data"]  - Normalization_values["min_data"])*f  + Normalization_values["min_data"]
       return Poisson_pde_loss(u=u.squeeze(1),f=f.squeeze(1),p=p,D=2)
    
    if which_example=='helmholtz':
       f=f+Normalization_values["mean_data"]
       u = (u*Normalization_values["std_model"])+ Normalization_values["mean_model"]
       return Helmholz_loss(u=u.squeeze(1),a=f[:,0,:,:],boundary=f[:,1,0,0],omega=5*torch.pi/2,p=p,D=1,Boundary_decay=Boundary_decay,pad_factor=pad_factor)
    
    if which_example=='wave_05':
       return Wave_pde_loss(u,f.squeeze,p=p,D=2)
    
def Operator_loss(output_train,output_fix,p=1):
    '''Calculates the PDE loss for the Poisson equation
       Input: output_trained : Output of network to fine tune, Shape = (Batch_size,Grid_size,Grid_size)
              output_fix : Output of network that was already pretrained
              p : Do we use L1 or L2 errors? Default: L1'''
    if p == 1:
       loss = torch.nn.L1Loss()
    elif p == 2:
       loss = torch.nn.MSELoss()
    elif p==3:
       loss = torch.nn.L1Loss()
    
    loss = loss(output_train, output_fix)
    return loss


def Poisson_pde_loss(u,f,p=1,D=2):
    '''Calculates the PDE loss for the Poisson equation
       Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
              f  Input  of network, Shape = (Batch_size,Grid_size,Grid_size)
              p  Do we use L1 or L2 errors? Default: L1
              D  Period of Domain
       Warning: Input f and Output u should not be normalized!'''
    
    Laplace_u=Laplace(u,D=D)

    if p == 1:
      loss = torch.nn.L1Loss()
    elif p == 2:
      loss = torch.nn.MSELoss()

    #Here I am not sure if I should use the relative loss
    loss_pde = loss(-Laplace_u, f)
    
    #Add boundary loss: u=0 on boundary(Domain)
    boundary_lossx=loss(u[:,0,:],torch.zeros_like(u[:,0,:]).to(u[:,0,:].device))
    boundary_lossy=loss(u[:,:,0],torch.zeros_like(u[:,:,0]).to(u[:,:,0].device))
    boundary_loss=0.5*(boundary_lossx+boundary_lossy)
    loss=0.5*(loss_pde+boundary_loss) #Here one could also choose a different tuning parameter

    return loss

def Helmholz_loss(u,a,boundary,omega=5*torch.pi/2,p=1,D=1,Boundary_decay=1,pad_factor=0):
    '''Calculates the PDE loss for the Poisson equation
    Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
           a  Input  of network, Shape = (Batch_size,Grid_size,Grid_size)
           boundary Boundary conditon of u at 0 and 1, float
           omega wave number
           p  Do we use L1 or L2 errors? Default: L1
           D  Period of Domain
           Boundary_decay hyperparemeter for boundary loss
    Warning: Input f and Output u should not be normalized!'''
    s=u.size(-1)
    center=s//2
    r=(s-2*pad_factor)//2
    l=1 #Take only interior
    if p == 1:
      loss = torch.nn.L1Loss()
      loss_b=loss
    elif p == 2:
      loss = torch.nn.MSELoss()
      loss_b=loss
    elif p==3:
      loss = torch.nn.SmoothL1Loss()
      loss_b = torch.nn.SmoothL1Loss()
    elif p==4:
      loss= L1_loss_weighted(in_shape=(s-l,s-l),device=u.device)
      loss_b = torch.nn.L1Loss()

    #Add boundary loss
    boundary=boundary.unsqueeze(-1)
    boundary_lossx_0=loss_b(u[:,pad_factor,:],  torch.mul(boundary,torch.ones_like(u[:,pad_factor,:])))
    boundary_lossy_0=loss_b(u[:,:,pad_factor],  torch.mul(boundary,torch.ones_like(u[:,:,pad_factor])))

    boundary_lossx_D=loss_b(u[:,-pad_factor-1,:], torch.mul(boundary,torch.ones_like(u[:,-pad_factor-1,:])))
    boundary_lossy_D=loss_b(u[:,:,-pad_factor-1], torch.mul(boundary,torch.ones_like(u[:,:,-pad_factor-1])))
    boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)

    

    reduce_gibbs_phenomenon=False
    if reduce_gibbs_phenomenon:
       boundary=boundary.squeeze(-1)
       u=torch.cat([u, -torch.flip(u,dims=[-1])[...,:,1:] +2*torch.ones_like(u)[...,:,1:]*boundary[:,None,None]],-1)
       u=torch.cat([u, -torch.flip(u,dims=[-2])[...,1:,:] +2*torch.ones_like(u)[...,1:,:]*boundary[:,None,None]],-2)
       u=u[...,:-1,:-1]
       D=D*2
       a=transform(a.unsqueeze(1)).squeeze(1)
       
    Laplace_u=Laplace(u,D=D)

    #loss_pde=loss(Laplace_u[:,l:-l,l:-l],-omega**2*a[:,l:-l,l:-l]**2*u[:,l:-l,l:-l])

    loss_pde=loss(Laplace_u[center-r:center+r,center-r:center+r],-omega**2*a[...,center-r:center+r,center-r:center+r]**2*u[...,center-r:center+r,center-r:center+r])

    loss=loss_pde+boundary_loss*Boundary_decay

    return loss

def Laplace(u,D=1):
    '''Calculates Laplace
    Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
    Ouput: \Delta u
           '''
    s=u.size(-1)
    
    #Calculate derivative using the fact FT(f')=2*pi*i*k*FT(f)
    u_hat=torch.fft.fft2(u,dim=[-2,-1])

    assert (u.device==u_hat.device) #Need to be same device, can only be checked on GPU

    k_max=s//2
    
    k_x = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                     torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(s, 1).repeat(1, s).reshape(1,s,s)
    k_y = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                     torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(1, s).repeat(s, 1).reshape(1,s,s)

    #Calculate Laplace of u
    Laplace_u_hat =-4*(torch.pi/D)**2*(k_x**2+k_y**2)*u_hat

    Laplace_u=torch.fft.irfft2(Laplace_u_hat[:, :, :k_max + 1], dim=[-2, -1])
    #Laplace_u=torch.fft.ifft2(Laplace_u_hat, dim=[-2, -1])

    return Laplace_u

def Laplace_inverse(u,D):
    s=u.size(-1)
    
    #Calculate derivative using the fact FT(f')=2*pi*i*k*FT(f)
    u_hat=torch.fft.fft2(u,dim=[-2,-1])

    assert (u.device==u_hat.device) #Need to be same device, can only be checked on GPU

    k_max=s//2
    
    k_x = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                     torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(s, 1).repeat(1, s).reshape(1,s,s)
    k_y = torch.cat((torch.arange(start=0, end=k_max+1, step=1, device=u.device),
                     torch.arange(start=-k_max+1, end=0, step=1, device=u.device)), 0).reshape(1, s).repeat(s, 1).reshape(1,s,s)
    
    Lap=-4*(torch.pi/D)**2*(k_x**2+k_y**2)
    Lap[0,0]=1
    #Calculate Laplace of u
    Laplace_inv_hat =1/Lap*u_hat

    Laplace_inv=torch.fft.irfft2(Laplace_inv_hat[:, :, :k_max + 1], dim=[-2, -1])
    return Laplace_inv


def Wave_pde_loss(u,u0,c=0.1,T=5,p=1,D=2):
    '''Calculates the PDE loss for the Wave equation
       Input:  u   Output of network, Shape = (Batch_size,Spatial_grid_size,Spatial_grid_size,Temporal_grid_size)
               u0  Input  of network, Shape = (Batch_size,Spatial_grid_size,Spatial_grid_size)
               p   Do we use L1 i.e (p==1) or L2 i.e (p==2) errors? Default: L1
               D  Period of Domain'''
   
    Temporal_grid_size=u.size(1)
    Spatial_grid_size=u.size(-1)
    #Calculate derivative using the fact FT(f')=2*pi*i*k*FT(f) (Where we use torch, Bogdan uses scypi)
    u_hat=torch.fft.fft2(u,dim=[-2,-1])

    assert (u.device==u_hat.device) #Need to be same device, can only be checked on GPU

    #Doesnt this make it a grid based loss? (No it shouldnt!)
    k_max=Spatial_grid_size//2
    dt=T/(Temporal_grid_size-1)
    
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(Spatial_grid_size, 1).repeat(1, Spatial_grid_size).reshape(1,1,Spatial_grid_size,Spatial_grid_size)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1, Spatial_grid_size).repeat(Spatial_grid_size, 1).reshape(1,1,Spatial_grid_size,Spatial_grid_size)
    
    ux_hat  = 2j *torch.pi*k_x*u_hat
    uxx_hat = 2j *torch.pi*k_x*ux_hat

    uy_hat  = 2j*torch.pi*k_y*u_hat
    uyy_hat = 2j*torch.pi*k_y*uy_hat
    
    uxx = torch.fft.irfftn(uxx_hat[:, :,:, :k_max + 1],   dim=[-2, -1])
    uyy = torch.fft.irfftn(uyy_hat[:, :,:, :k_max + 1],   dim=[-2, -1])

    #Calculate Laplace-Operator for u
    Du=(uyy+uxx)/D**2 #Divide by the period of the function
   
    utt = (u[:, 2:,...] - 2.0*u[:, 1:-1,...] + u[:, :-2,...]) / (dt**2)

    if p == 1:
      loss = torch.nn.L1Loss()
    elif p == 2:
      loss = torch.nn.MSELoss()
    
    epsilon=1e-10

    #Here I am not sure if I should use the relative loss
    loss_pde = loss(Du[:, 1:-1,:],c**2* utt) / (loss(torch.zeros_like(u0).to(u0.device), u0)+epsilon)
    
    #Add relative boundary loss: u=0 on boundary(Domain)
    inital_loss=loss(u0, u[:,0,...])/(loss(torch.zeros_like(u0).to(u0.device), u0)+epsilon)

    return 0.5*(loss_pde+inital_loss)
