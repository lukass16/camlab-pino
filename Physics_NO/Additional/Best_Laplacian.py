import torch
import numpy as np
import h5py
import sys
sys.path.append('..')
from loss_functions.Pde_loss import Laplace
import matplotlib.pyplot as plt

def Helmholz_loss(u,a,boundary,omega=5*torch.pi/2,p=1,D=1,Boundary_decay=1):
    '''Calculates the PDE loss for the Poisson equation
    Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
           a  Input  of network, Shape = (Batch_size,Grid_size,Grid_size)
           boundary Boundary conditon of u at 0 and 1, float
           omega wave number
           p  Do we use L1 or L2 errors? Default: L1
           D  Period of Domain
           Boundary_decay hyperparemeter for boundary loss
    Warning: Input f and Output u should not be normalized!'''

    Laplace_u=Laplace(u,D=D)
    #Laplace_inverse_a=Laplace_inverse(-omega**2*a**2*u,D)
    s=u.size(-1)

    l=1 #Take only interior
    if p == 1:
      loss = torch.nn.L1Loss()
      loss_b=loss
    elif p == 2:
      loss = torch.nn.MSELoss()
      loss_b=loss
    
    #loss_pde=loss(Laplace_u[:,l:-l,l:-l],-omega**2*a[:,l:-l,l:-l]**2*u[:,l:-l,l:-l])

    loss_pde=loss(Laplace_u,-omega**2*a**2*u)
    #loss_pde=loss(Laplace_inverse_a,u)
    if s==128+32:
       k=16
       loss_pde=loss(Laplace_u[...,k:-k,k:-k],-omega**2*a[...,k:-k,k:-k]**2*u[...,k:-k,k:-k])
 
    if s==256:
       loss_pde=loss(Laplace_u[...,0:128,0:128],-omega**2*a[...,0:128,0:128]**2*u[...,0:128,0:128])
    if s==256+50:
       m=128+25
       loss_pde=loss(Laplace_u[...,0:m,0:m],-omega**2*a[...,0:m,0:m]**2*u[...,0:m,0:m])
      

    #Add boundary loss
    boundary=boundary.unsqueeze(-1)
    boundary_lossx_0=loss_b(u[:,0,:],  torch.mul(boundary,torch.ones_like(u[:,0,:])))
    boundary_lossx_D=loss_b(u[:,-1,:], torch.mul(boundary,torch.ones_like(u[:,-1,:])))
    boundary_lossy_D=loss_b(u[:,:,-1], torch.mul(boundary,torch.ones_like(u[:,:,-1])))
    boundary_lossy_0=loss_b(u[:,:,0],  torch.mul(boundary,torch.ones_like(u[:,:,0])))
    boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
    
    boundary_loss=0
    loss=loss_pde+boundary_loss*Boundary_decay

    return loss

def Helmholz_loss3(u,a,boundary,omega=5*torch.pi/2,p=1,D=1,Boundary_decay=1):
    '''Calculates the PDE loss for the Poisson equation
    Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
           a  Input  of network, Shape = (Batch_size,Grid_size,Grid_size)
           boundary Boundary conditon of u at 0 and 1, float
           omega wave number
           p  Do we use L1 or L2 errors? Default: L1
           D  Period of Domain
           Boundary_decay hyperparemeter for boundary loss
    Warning: Input f and Output u should not be normalized!'''

    l=1 #Take only interior
    if p == 1:
      loss = torch.nn.L1Loss()
      loss_b=loss
    elif p == 2:
      loss = torch.nn.MSELoss()
      loss_b=loss

    #Add boundary loss
    boundary=boundary.unsqueeze(-1)
    boundary_lossx_0=loss_b(u[:,0,:],  torch.mul(boundary,torch.ones_like(u[:,0,:])))
    boundary_lossy_0=loss_b(u[:,:,0],  torch.mul(boundary,torch.ones_like(u[:,:,0])))

    boundary_lossx_D=loss_b(u[:,-1,:], torch.mul(boundary,torch.ones_like(u[:,-1,:])))
    boundary_lossy_D=loss_b(u[:,:,-1], torch.mul(boundary,torch.ones_like(u[:,:,-1])))
    boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)

    s=u.size(-1)

    reduce_gibbs_phenomenon=True
    if reduce_gibbs_phenomenon:
       boundary=boundary.squeeze(-1)
       u=torch.cat([u, -torch.flip(u,dims=[-1])[...,:,1:] +2*torch.ones_like(u)[...,:,1:]*boundary[:,None,None]],-1)
       u=torch.cat([u, -torch.flip(u,dims=[-2])[...,1:,:] +2*torch.ones_like(u)[...,1:,:]*boundary[:,None,None]],-2)
       u=u[...,:-1,:-1]
       D=D*2
       a=transform(a.unsqueeze(1)).squeeze(1)
       
    Laplace_u=Laplace(u,D=D)

    #loss_pde=loss(Laplace_u[:,l:-l,l:-l],-omega**2*a[:,l:-l,l:-l]**2*u[:,l:-l,l:-l])

    loss_pde=loss(Laplace_u[...,:s,:s],-omega**2*a**2*u[...,0:s,0:s])

    loss=loss_pde+boundary_loss*Boundary_decay

    return loss
def Helmholz_loss2(u,a,boundary,omega=5*torch.pi/2,p=1,D=1,Boundary_decay=1):
    '''Calculates the PDE loss for the Poisson equation
    Input: u  Output of network, Shape = (Batch_size,Grid_size,Grid_size)
           a  Input  of network, Shape = (Batch_size,Grid_size,Grid_size)
           boundary Boundary conditon of u at 0 and 1, float
           omega wave number
           p  Do we use L1 or L2 errors? Default: L1
           D  Period of Domain
           Boundary_decay hyperparemeter for boundary loss
    Warning: Input f and Output u should not be normalized!'''

    Laplace_u=Laplace(u,D=D)
    #Laplace_inverse_a=Laplace_inverse(-omega**2*a**2*u,D)
    s=u.size(-1)

    l=1 #Take only interior
    if p == 1:
      loss = torch.nn.L1Loss()
      loss_b=loss
    elif p == 2:
      loss = torch.nn.MSELoss()
      loss_b=loss
    
    #loss_pde=loss(Laplace_u[:,l:-l,l:-l],-omega**2*a[:,l:-l,l:-l]**2*u[:,l:-l,l:-l])
    loss_pde=loss(Laplace_u[...,0:128,0:128],-omega**2*a**2*u[...,0:128,0:128])
    #loss_pde=loss(Laplace_u,-omega**2*a**2*u)
    #loss_pde=loss(Laplace_inverse_a,u)
      

    #Add boundary loss
    boundary=boundary.unsqueeze(-1)
    boundary_lossx_0=loss_b(u[:,0,:],  torch.mul(boundary,torch.ones_like(u[:,0,:])))
    boundary_lossy_0=loss_b(u[:,:,0],  torch.mul(boundary,torch.ones_like(u[:,:,0])))
    boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0)
    loss=loss_pde+boundary_loss*Boundary_decay

    return loss

def transform(input):
        s=input.shape[-1]
        input=torch.nn.functional.interpolate(input, size=(s+1,s+1),mode='bilinear')
        input=input[...,:-1,:-1]
        return input

def transform_extend(input):
        s=input.shape[-1]
        input=torch.nn.functional.interpolate(input, size=(s+2,s+2),mode='bilinear')
        input=input[...,:-1,:-1]
        return input

def transform_higher(input,l=200):
        s=input.shape[-1]
        input=torch.nn.functional.interpolate(input, size=(l,l),mode='bilinear')
        return input
def transform_smooth(input,s):
        input=torch.nn.functional.interpolate(input, size=(s,s),mode='bilinear')
        return input

file_data='/cluster/scratch/harno/data/HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5'
reader = h5py.File(file_data, 'r')

n=100
input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:]).unsqueeze(0)
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:]).unsqueeze(0)
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
loss=Helmholz_loss(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=1)
x1 = torch.linspace(0, 1, 128)
y1 = torch.linspace(0, 1, 128)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[0,:,:].numpy(), cmap='viridis')
ax1.set_title('Normal',fontsize=20)
Lap_u=Laplace(output,1)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test_norm')
print('Helmoltz loss normal: ', loss)


import torchvision
#Filter=torchvision.transforms.GaussianBlur(3,1)
#input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:]).unsqueeze(0)
#output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
#boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
#down=-torch.flip(output,dims=[-1])[:,1:16]+2*torch.ones_like(output)*boundary
#up=torch.flip(output,dims=[-1]-torch.ones_like(output)[:,1:16]*down[:,16])[:,1:16]
#smooth_part=torch.cat([down,],-1)
#
##smooth_part = Filter(smooth_part.unsqueeze(0).unsqueeze(0))
##smooth_part = (smooth_part+2*torch.ones_like(smooth_part)*boundary).squeeze(0).squeeze(0)
#output=torch.cat([output,smooth_part],-1)
#smooth_part2=torch.cat([-torch.flip(output,dims=[-2])[1:16,:],torch.flip(output,dims=[-2])[1:16,:]],-2)
##smooth_part2 = Filter(smooth_part2.unsqueeze(0).unsqueeze(0))
##smooth_part2 = (smooth_part2+2*torch.ones_like(smooth_part2)*boundary).squeeze(0).squeeze(0)
#output=torch.cat([output,smooth_part2],-2).unsqueeze(0)
#loss=Helmholz_loss2(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=2)
#print('Helmoltz loss extended short: ', loss)
#plt.figure()
#plt.plot(np.arange(len(output[0,10,:])),output[0,10,:])
#plt.plot(np.arange(len(output[0,10,:])),output[0,:,10])
#output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
#plt.plot(np.arange(len(output[10,:])),output[10,:])
#plt.savefig('hey')


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
output=torch.cat([output,-torch.flip(output,dims=[-1])[:,1:]+2*torch.ones_like(output)[:,1:]*boundary],-1)
output=torch.cat([output, -torch.flip(output,dims=[-2])[1:,:]+2*torch.ones_like(output)[1:,:]*boundary],-2)
#input=torch.cat([input,-torch.flip(input,dims=[-1,-2])[1:,:]+2*torch.ones_like(input[1:,:])*input[0,0]],-2)
#input=torch.cat([input, torch.flip(input,dims=[-2,-1])[:,1:]],-1)
#output=torch.cat([output,-torch.flip(output,dims=[-1,-2])[1:,:]+2*torch.ones_like(output)[1:,:]*boundary],-2)
#output=torch.cat([output, torch.flip(output,dims=[-2,-1])[:,1:]],-1)
input=transform(input.unsqueeze(0).unsqueeze(0)).squeeze(0)
output=transform_extend(output.unsqueeze(0).unsqueeze(0)).squeeze(0) 
x1 = torch.linspace(0, 2, 256)
y1 = torch.linspace(0, 2, 256)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,3,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[0,:,:].numpy(), cmap='viridis')
ax1.set_title('Extended and transformed',fontsize=20)
Lap_u=Laplace(output,2)
ax2=fig.add_subplot(1,3,2, projection='3d')
ax2.plot_surface(x[:128,:128].numpy(), y[:128,:128].numpy(), Lap_u[0,:128,:128].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
ax3=fig.add_subplot(1,3,3, projection='3d')
ax3.plot_surface(x[:128,:128].numpy(), y[:128,:128].numpy(),-((5*torch.pi/2)**2*input[0,:128,:128]**2*output[0,:128,:128]).numpy(), cmap='viridis')
ax3.set_title('Real Laplace',fontsize=20)
plt.savefig('test1')
loss=Helmholz_loss2(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=2)
print('Helmoltz loss extended and transformed: ', loss)


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:]).unsqueeze(0)
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:]).unsqueeze(0)
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
input=transform(input.unsqueeze(0)).squeeze(0)
output=transform(output.unsqueeze(0)).squeeze(0) 
x1 = torch.linspace(0, 1, 128)
y1 = torch.linspace(0, 1, 128)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[0,:,:].numpy(), cmap='viridis')
ax1.set_title('Transformed',fontsize=20)
Lap_u=Laplace(output,1)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test_trans')
loss=Helmholz_loss(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=1)
print('Helmoltz loss transformed: ', loss)


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
shape=(128+32,128+32)
center=shape[0]//2
r=input.shape[0]//2
input_pad=torch.zeros(shape)
input_pad[center-r:center+r,center-r:center+r]=input
output_pad=torch.zeros(shape)
output_pad[center-r:center+r,center-r:center+r]=output
x1 = torch.linspace(0, 1, shape[0])
y1 = torch.linspace(0, 1, shape[0])
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output_pad[:,:].numpy(), cmap='viridis')
ax1.set_title('Zeros padded', fontsize=20)
Lap_u=Laplace(output_pad.unsqueeze(0),1+32/128)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test2')
loss=Helmholz_loss(u=output_pad.unsqueeze(0),a=input_pad.unsqueeze(0),boundary=boundary,omega=5*torch.pi/2,p=1,D=1+32/128)
print('Helmoltz loss zero_padded: ', loss)

input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
shape=(128+32,128+32)
center=shape[0]//2
r=input.shape[0]//2
input_pad=torch.ones(shape)*input[0,0]
input_pad[center-r:center+r,center-r:center+r]=input
output_pad=torch.ones(shape)*boundary
output_pad[center-r:center+r,center-r:center+r]=output
x1 = torch.linspace(0, 1, shape[0])
y1 = torch.linspace(0, 1, shape[0])
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,3,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output_pad[:,:].numpy(), cmap='viridis')
ax1.set_title('Boundary padded', fontsize=20)
Lap_u=Laplace(output_pad.unsqueeze(0),1+32/128)
ax2=fig.add_subplot(1,3,2, projection='3d')
ax2.plot_surface(x[16:-16,16:-16].numpy(), y[16:-16,16:-16].numpy(), Lap_u[0,16:-16,16:-16].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
ax3=fig.add_subplot(1,3,3, projection='3d')
ax3.plot_surface(x[16:-16,16:-16].numpy(), y[16:-16,16:-16].numpy(),-((5*torch.pi/2)**2*input**2*output).numpy(), cmap='viridis')
ax3.set_title('Real Laplace',fontsize=20)
plt.savefig('test3')
loss=Helmholz_loss(u=output_pad.unsqueeze(0),a=input_pad.unsqueeze(0),boundary=boundary,omega=5*torch.pi/2,p=1,D=1+32/128)
print('Helmoltz loss boundary_padded: ', loss)


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
input=transform(input.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
output=transform(output.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
input=torch.cat([input,-torch.flip(input,dims=[-1])+2*torch.ones_like(input)*input[0,0]],-1)
input=torch.cat([input,-torch.flip(input,dims=[-2])+2*torch.ones_like(input)*input[0,0]],-2)
output=torch.cat([output,-torch.flip(output,dims=[-1])+2*torch.ones_like(output)*boundary],-1)
output=torch.cat([output, -torch.flip(output,dims=[-2])+2*torch.ones_like(output)*boundary],-2)
x1 = torch.linspace(0, 2, 256)
y1 = torch.linspace(0, 2, 256)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[:,:].numpy(), cmap='viridis')
ax1.set_title('Interpolate then extended', fontsize=20)
Lap_u=Laplace(output.unsqueeze(0),2)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test4')
loss=Helmholz_loss(u=output.unsqueeze(0),a=input.unsqueeze(0),boundary=boundary,omega=5*torch.pi/2,p=1,D=2)
print('Helmoltz loss Interpolate then extended: ', loss)


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
input=transform_higher(input.unsqueeze(0).unsqueeze(0),l=200).squeeze(0).squeeze(0)
output=transform_higher(output.unsqueeze(0).unsqueeze(0),l=200).squeeze(0).squeeze(0)
x1 = torch.linspace(0, 2, 200)
y1 = torch.linspace(0, 2, 200)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[:,:].numpy(), cmap='viridis')
ax1.set_title('Higher grid', fontsize=20)
Lap_u=Laplace(output.unsqueeze(0),1)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test5')
loss=Helmholz_loss(u=output.unsqueeze(0),a=input.unsqueeze(0),boundary=boundary,omega=5*torch.pi/2,p=1,D=1)
print('Helmoltz loss finer grid: ', loss)

input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
input=transform_higher(input.unsqueeze(0).unsqueeze(0),l=200)
output=transform_higher(output.unsqueeze(0).unsqueeze(0),l=200)
input=transform(input).squeeze(0).squeeze(0)
output=transform(output).squeeze(0).squeeze(0)
x1 = torch.linspace(0, 1, 200)
y1 = torch.linspace(0, 1, 200)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[:,:].numpy(), cmap='viridis')
ax1.set_title('Higher grid', fontsize=20)
Lap_u=Laplace(output.unsqueeze(0),1)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test6')
loss=Helmholz_loss(u=output.unsqueeze(0),a=input.unsqueeze(0),boundary=boundary,omega=5*torch.pi/2,p=1,D=1)
print('Helmoltz loss finer grid and transform: ', loss)


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
input=torch.cat([input,-torch.flip(input,dims=[-1])[:,1:]+2*torch.ones_like(input[:,1:])*input[0,0]],-1)
input=torch.cat([input,-torch.flip(input,dims=[-2])[1:,:]+2*torch.ones_like(input[1:,:])*input[0,0]],-2)
output=torch.cat([output,-torch.flip(output,dims=[-1])[:,1:]+2*torch.ones_like(output)[:,1:]*boundary],-1)
output=torch.cat([output, -torch.flip(output,dims=[-2])[1:,:]+2*torch.ones_like(output)[1:,:]*boundary],-2)
input=transform_higher(input.unsqueeze(0).unsqueeze(0),l=256+50).squeeze(0)
output=transform_higher(output.unsqueeze(0).unsqueeze(0),l=256+50).squeeze(0) 
x1 = torch.linspace(0, 2, 256+50)
y1 = torch.linspace(0, 2, 256+50)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,3,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[0,:,:].numpy(), cmap='viridis')
ax1.set_title('Extended and transformed',fontsize=20)
Lap_u=Laplace(output,2)
ax2=fig.add_subplot(1,3,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
ax3=fig.add_subplot(1,3,3, projection='3d')
ax3.plot_surface(x[:128,:128].numpy(), y[:128,:128].numpy(),-((5*torch.pi/2)**2*input[0,:128,:128]**2*output[0,:128,:128]).numpy(), cmap='viridis')
ax3.set_title('Real Laplace',fontsize=20)
plt.savefig('test7')
loss=Helmholz_loss(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=2)
print('Helmoltz loss extended and transformed and higher: ', loss)


input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
input=transform(input.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
output=transform(output.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
input=torch.cat([input,-torch.flip(input,dims=[-1])+2*torch.ones_like(input)*input[0,0]],-1)
input=torch.cat([input,-torch.flip(input,dims=[-2])+2*torch.ones_like(input)*input[0,0]],-2)
output=torch.cat([output,-torch.flip(output,dims=[-1])+2*torch.ones_like(output)*boundary],-1)
output=torch.cat([output, -torch.flip(output,dims=[-2])+2*torch.ones_like(output)*boundary],-2)
input=transform_smooth(input.unsqueeze(0).unsqueeze(0),s=256).squeeze(0).squeeze(0)
output=transform_smooth(output.unsqueeze(0).unsqueeze(0),s=256).squeeze(0).squeeze(0)
x1 = torch.linspace(0, 2, 256)
y1 = torch.linspace(0, 2, 256)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[:,:].numpy(), cmap='viridis')
ax1.set_title('Interpolate then extended and interpolate', fontsize=20)
Lap_u=Laplace(output.unsqueeze(0),2)
ax2=fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(x.numpy(), y.numpy(), Lap_u[0,:,:].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
plt.savefig('test8')
loss=Helmholz_loss(u=output.unsqueeze(0),a=input.unsqueeze(0),boundary=boundary,omega=5*torch.pi/2,p=1,D=2)
print('Helmoltz loss Interpolate then extended interpolate ', loss)



#For test and validation best can also take away the interpolation
input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
output=torch.cat([output,-torch.flip(output,dims=[-1])[:,1:]+2*torch.ones_like(output)[:,1:]*boundary],-1)
output=torch.cat([output, -torch.flip(output,dims=[-2])[1:,:]+2*torch.ones_like(output)[1:,:]*boundary],-2)
input=transform(input.unsqueeze(0).unsqueeze(0)).squeeze(0)
output=output[:-1,:-1].unsqueeze(0)
#output=transform_smooth(output.unsqueeze(0),s=256).squeeze(0)
x1 = torch.linspace(0, 2, 254)
y1 = torch.linspace(0, 2, 254)
x, y = torch.meshgrid(x1, y1,indexing='ij')
fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(1,3,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), output[0,:,:].numpy(), cmap='viridis')
ax1.set_title('Extended and transformed',fontsize=20)
Lap_u=Laplace(output,2)
ax2=fig.add_subplot(1,3,2, projection='3d')
ax2.plot_surface(x[:128,:128].numpy(), y[:128,:128].numpy(), Lap_u[0,:128,:128].numpy(), cmap='viridis')
ax2.set_title('Laplace',fontsize=20)
ax3=fig.add_subplot(1,3,3, projection='3d')
ax3.plot_surface(x[:128,:128].numpy(), y[:128,:128].numpy(),-((5*torch.pi/2)**2*input[0,:128,:128]**2*output[0,:128,:128]).numpy(), cmap='viridis')
ax3.set_title('Real Laplace',fontsize=20)
plt.savefig('test9')
loss=Helmholz_loss2(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=2)
print('Helmoltz loss extended: ', loss)


#Test helmholtz3
input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:])
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:])
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
loss=Helmholz_loss3(u=output.unsqueeze(0),a=input.unsqueeze(0),boundary=boundary.unsqueeze(0),omega=5*torch.pi/2,p=1,D=1)
print('helmholtz3 loss',loss)





