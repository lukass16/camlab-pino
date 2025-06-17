import torch
import numpy as np

inputs=torch.randn((1,3,3))
s=10
a=1
b     =  torch.from_numpy(np.array(a)).to(torch.float32)
bc    =  torch.ones_like(inputs)
bc[...,0,:]  = b
bc[...,-1,:] = b
bc[...,:,0]  = b
bc[...,:,-1] = b
inputs1 = torch.cat((inputs, bc), 0)
inputs=torch.randn((1,3,3))
a=2
b     =  torch.from_numpy(np.array(a)).to(torch.float32)
bc    =  torch.zeros_like(inputs)
bc[...,0,:]  = b
bc[...,-1,:] = b
bc[...,:,0]  = b
bc[...,:,-1] = b
inputs2 = torch.cat((inputs, bc), 0)
inputs=torch.cat((inputs1.unsqueeze(0),inputs2.unsqueeze(0)),0)
boundary=inputs[:,1,0,0]
import h5py
file_to_data='/cluster/scratch/harno/data/HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5'
n=np.random.randint(200)
reader = h5py.File(file_to_data, 'r')
input= torch.from_numpy(reader['Sample_' + str(n)]["a"][:]).to(torch.float32).unsqueeze(0)
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:]).to(torch.float32).unsqueeze(0)
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)

input1= torch.from_numpy(reader['Sample_' + str(n+1)]["a"][:]).to(torch.float32).unsqueeze(0)
output1=torch.from_numpy(reader['Sample_' + str(n+1)]["u"][:]).to(torch.float32).unsqueeze(0)
boundary1=torch.from_numpy(np.array(reader['Sample_' + str(n+1)]["bc"])).to(torch.float32)

outputs=torch.cat((output.unsqueeze(0),output1.unsqueeze(0)),0)
boundarys=torch.cat((boundary.unsqueeze(0),boundary1.unsqueeze(0)),0)

outputs=outputs.squeeze(1)
print(boundarys.size(),(torch.mul(boundarys.unsqueeze(-1),torch.ones_like(outputs[...,:,-1]))).size())
loss = torch.nn.L1Loss()
multbound1=loss(outputs[:,:,-1],torch.mul(boundarys.unsqueeze(-1),torch.ones_like(outputs[...,:,-1])))
multbound=loss(outputs[:,:,0] ,torch.mul(boundarys.unsqueeze(-1),torch.ones_like(outputs[...,:,-1])))
print(multbound,multbound1)

x=torch.ones((3,2,2))
z=torch.tensor([1,2,3])
b=x*z[:,None,None]
#print(b)

def transform(input):
        s=input.shape[-1]
        input=torch.nn.functional.interpolate(input, size=(s+1,s+1),mode='bicubic')
        input=input[...,:-1,:-1]
        return input

import h5py
import sys
sys.path.append('..')
from loss_functions.Pde_loss import Helmholz_loss,Laplace
import matplotlib.pyplot as plt
#file_data='/cluster/scratch/harno/data/HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5'
#reader = h5py.File(file_data, 'r')
#n=100
#input=torch.from_numpy(reader['Sample_' + str(n)]["a"][:]).unsqueeze(0)
#output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:]).unsqueeze(0)
#boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
#input=torch.cat([input,-torch.flip(input,dims=[-1,-2])+2*torch.ones_like(input)*boundary],-2)
#input=torch.cat([input, torch.flip(input,dims=[-2,-1])],-1)
#output=torch.cat([output,-torch.flip(output[1:,1:],dims=[-1,-2])+2*torch.ones_like(output)*boundary],-2)
#output=torch.cat([output, torch.flip(output[1:,1:],dims=[-2,-1])],-1)
#input=transform(input.unsqueeze(0)).squeeze(0)
#output=transform(output.unsqueeze(0)).squeeze(0) 
#print(input.shape,output.shape)
#
##u=torch.cat([u,-torch.flip(u,dims=[-1,-2])+2*torch.ones_like(u)*boundary],-2)
##u=torch.cat([u, torch.flip(u,dims=[-2,-1])],-1)
#loss=Helmholz_loss(u=output,a=input,boundary=boundary,omega=5*torch.pi/2,p=1,D=1)
#
#u=output
#print(loss)

x=torch.randn((16,128,128))
z=torch.randn((16,128,128))
x=abs(x-z)
y=torch.randn((128,128))
b=y[None,...]*x
print(b.shape)
#
#u=Laplace(u,2)
#print(boundary)
#flip=-torch.flip(u,dims=[-1,-2])+2*torch.ones_like(u)*boundary
#u=torch.cat([u,flip],1)
#x1 = torch.linspace(0, 2, 256)
#y1 = torch.linspace(0, 2, 256)
#x, y = torch.meshgrid(x1, y1)
#fig = plt.figure(figsize=(15, 15))
#ax1=fig.add_subplot(1,1,1, projection='3d')
#ax1.plot_surface(x.numpy(), y.numpy(), u[0,:,:].numpy(), cmap='viridis')
#plt.savefig('hey')
#print(loss)
#plt.plot(np.arange(0,u.size(-2)),u[0,:,10])
#plt.plot(np.arange(0,u.size(-2)),u[0,10,:])
def Laplace_hat(u,D=1):
    s=u.size(-1)
    u_hat=torch.fft.fft2(u,dim=[-2,-1])

    assert (u.device==u_hat.device) #Need to be same device, can only be checked on GPU

    k_max=s//2
    
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(s, 1).repeat(1, s).reshape(1,s,s)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1, s).repeat(s, 1).reshape(1,s,s)
    
    #Calculate Laplace of u
    Laplace_u_hat =-4*(torch.pi/D)**2*(k_x**2+k_y**2)*u_hat
    return Laplace_u_hat



#s=128
#x1 = torch.linspace(0, D,s)
#y1 = torch.linspace(0, D,s)
#x, y = torch.meshgrid(x1,y1)
#frequencies_input_x = torch.fft.fftfreq(input.shape[0], 1.0 / (x1[1].item() - x1[0].item()))
#frequencies_input_y = torch.fft.fftfreq(input.shape[1], 1.0 / (y1[1].item() - y1[0].item())) 
#frequencies_x, frequencies_y = torch.meshgrid(frequencies_input_x ,frequencies_input_y )
#Laplace_u=Laplace(u,D).squeeze(0)
#
#fig = plt.figure(figsize=(15, 15))
#fft_result = Laplace_hat(output,D).squeeze(0)
#fft_result_shifted_Laplace_u = torch.fft.fftshift(fft_result)   
#ax1=fig.add_subplot(1,2,1, projection='3d')
#ax1.plot_surface(x.numpy(), y.numpy(), Laplace_u.numpy(), cmap='viridis')
#ax1.set_title('Laplace')
#ax2 = fig.add_subplot(1,2,2)
#ax2.imshow(torch.abs(fft_result_shifted_Laplace_u).numpy(), extent=(frequencies_x.min(), frequencies_x.max(), frequencies_y.min(), frequencies_y.max()), origin='lower', cmap='viridis')
#ax2.set_title('Laplace Magnitude Spectrum')
#ax2.set_xlabel('Frequency (u)')
#ax2.set_ylabel('Frequency (v)')
#plt.tight_layout()
#plt.savefig('hey2')


#alising=False
#if alising:
#   u=torch.cat([u,-torch.flip(u,dims=[-1,-2])+2*torch.ones_like(u)*boundary],-2)
#   u=torch.cat([u,torch.flip(u,dims=[-1,-2])],-1)
#   a=torch.cat([a,-torch.flip(a,dims=[-1,-2])+2*torch.ones_like(a)*a[...,:,0]],-2)
#   a=torch.cat([a,torch.flip(a,dims=[-1,-2])],-1)
#   D=D*2


#filter_rate=0
#if  filter_rate !=0:
#    Laplace_u_hat_shifted = torch.fft.fftshift(Laplace_u_hat,dim=[-2,-1]) 
#    c_x,c_y=s//2,s//2   
#    r_x,r_y =4,4
#    mask=torch.zeros_like(Laplace_u_hat_shifted)
#    mask[:,c_x-r_x:c_x+r_x,c_y-r_y:c_y+r_y]=1
#    Laplace_u_hat_shifted= Laplace_u_hat_shifted * mask
#    Laplace_u_hat = torch.fft.ifftshift(Laplace_u_hat_shifted,dim=[-2,-1])


#def Laplace_finit(u,D):
#    s=u.size(-1)
#    Laplace_u=(u[...,:-2,1:-1]+u[...,2:,1:-1]+u[...,1:-1,:-2]+u[...,1:-1,2:]+4*u[...,1:-1,1:-1])*(D/s-1)
#    return Laplace_u




#k_max=64
#s=2*k_max
#k_y = torch.cat((torch.arange(start=0, end=k_max+1, step=1),
#                     torch.arange(start=-k_max+1, end=0, step=1)), 0).reshape(1, s).repeat(s, 1).reshape(1,s,s)
#k_x = torch.cat((torch.arange(start=0, end=k_max+1, step=1),
#                     torch.arange(start=-k_max+1, end=0, step=1)), 0).reshape(s, 1).repeat(1, s).reshape(1,s,s)
#w=torch.ones(s,s)
#u_hat=torch.fft.fft2(w,dim=[-2,-1],s=(s,s))
#u_hat=4*(torch.pi/D)**2*(k_x**2+k_y**2)*u_hat
#u_hat_in=torch.fft.irfft2(w[:,:k_max+1],dim=[-2,-1])
#u_hat_in_l=torch.fft.ifft2(w,dim=[-2,-1])
#print(u_hat_in.shape,u_hat_in_l.shape)
#s=128