import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import sys
sys.path.append("../..")
sys.path.append("..")
import torch.nn.functional as F
from training.FourierFeatures import FourierFeaturesPrecondition
from loss_functions.ModulePDELoss import Laplace,Loss_PDE
from Old_files.Pde_loss import Poisson_pde_loss,Laplace,Helmholz_loss

def plot_image(input,output,name):
     cmap = "gist_ncar"
     fig,axes=plt.subplots(1,2)
     axes[0].imshow(input.T, cmap=cmap)
     axes[1].imshow(output.T, cmap=cmap)
     axes[0].set_title('Input')
     axes[1].set_title('Output')
     plt.savefig(name)

def plot_Fourier(input,F_n,name):
     cmap = "gist_ncar"
     input=torch.from_numpy(input).to(torch.float32)
     fig=plt.figure()
     FF=FourierFeaturesPrecondition(F_n,device='cpu')
     FF_input=FF(input)
     fig,axes=plt.subplots(1,2)
     axes[0].imshow(input.T, cmap=cmap)
     axes[1].imshow(FF_input.T, cmap=cmap)
     axes[0].set_title('Input')
     axes[1].set_title('Fourier')
     plt.savefig(name)

which_example='helmholtz'
s=128
D=1

N=100
n=np.random.randint(0,N)

if which_example=='poisson':
   file_data='/cluster/scratch/harno/data/PoissonData_PDEDomain2.h5'
   file_data_out='/cluster/scratch/harno/data/PoissonData_PDE_outDomain2.h5'
   input_str="input"
   output_str="output"
   reader = h5py.File(file_data, 'r')
   input=reader['Sample_' + str(n)][input_str][:]
   output=reader['Sample_' + str(n)][output_str][:]
   print(f'Input shape: {input.shape}')
   print(f'Output shape: {output.shape}')
   torch.from_numpy(output).unsqueeze(0)
   loss=Poisson_pde_loss(u=torch.from_numpy(output).unsqueeze(0),f=torch.from_numpy(input).unsqueeze(0),p=2,D=2)
   print(loss)


elif which_example=='helmholtz':
    #file_data='/cluster/scratch/harno/data/HelmotzData_FixedBC1_4shapes_fixed_w_processed_2.h5'
    file_data='/cluster/scratch/harno/data/HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5'
    reader = h5py.File(file_data, 'r')
    input=reader['Sample_' + str(n)]["a"][:]
    output=reader['Sample_' + str(n)]["u"][:]
    boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)

    print(f'Input shape: {input.shape}')
    print(f'Output shape: {output.shape}')
    print(f'boundary: {boundary}')
    #input_cat = torch.cat((torch.from_numpy(input).unsqueeze(0),(boundary*torch.ones_like(torch.from_numpy(input))).unsqueeze(0)), 0)
    #print(f'Cat input shape: {input_cat.shape}')
    #print(f'Boundary of output interior: {output[0,:]}, {output[-1,:]}, {output[:,0]}, {output[:,-1]}')
    omega=5*torch.pi/2
    p=3
    loss=Helmholz_loss(u=torch.from_numpy(output).unsqueeze(0),a=torch.from_numpy(input).unsqueeze(0),boundary=boundary,omega=omega,p=p,D=1)
    print(loss)
    Laplace_u=Laplace(u=torch.from_numpy(output).unsqueeze(0),D=1)
    im=Laplace_u+omega**2*torch.from_numpy(input).unsqueeze(0)*torch.from_numpy(output).unsqueeze(0)
 
    test=(-omega**2*torch.from_numpy(input).unsqueeze(0)**2*torch.from_numpy(output).unsqueeze(0))
    plt.figure()
    fig,ax=plt.subplots(1,3)
    im1=ax[0].imshow(torch.from_numpy(output).T, cmap="gist_ncar",label='u')
    im2=ax[1].imshow(Laplace_u[0,2:-2,2:-2].T, cmap="gist_ncar",label='Laplace')
    im3=ax[2].imshow(test.permute(*torch.arange(test.ndim - 1, -1, -1)), cmap="gist_ncar",label='a**2*u')
    fig.colorbar(im1,ax=ax[0])
    fig.colorbar(im2,ax=ax[1])
    fig.colorbar(im3,ax=ax[2])
    ax[0].set_title('u')
    ax[1].set_title('Laplace')
    ax[2].set_title('a**2*u')
    plt.savefig('Laplace')
    
    plt.figure()
    plt.plot(np.arange(len(Laplace_u[0,0,:])),Laplace_u[0,10,:],label='lap')
    plt.plot(np.arange(len(Laplace_u[0,0,:])),     test[0,10,:],label='cor')
    plt.plot(np.arange(len(Laplace_u[0,0,:])),torch.from_numpy(output)[10,:],label='u')
    plt.legend()
    plt.savefig('Check Oscillation')
    print(loss)

plot_image(input,output,f'../Plots/{which_example}')


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
    Laplace_u_hat =-4*(torch.pi/D)**2*(k_x**2+k_y**2)
    Laplace_u_hat =Laplace_u_hat*u_hat
    return Laplace_u_hat

#plot_Fourier(input,6,f'../Plots/Fourier{which_example}')
#
#reader = h5py.File(file_data_out, 'r')
#input=reader['Sample_' + str(n)][input_str][:]
#output=reader['Sample_' + str(n)][output_str][:]
#plot_image(input,output,f'../Plots/{which_example}_out')

x1 = torch.linspace(0, D, s)
y1 = torch.linspace(0, D, s)
x, y = torch.meshgrid(x1, y1, indexing='ij')

print('Check Spectrum')
input=torch.from_numpy(input)
fft_result = torch.fft.fft2(input)
fft_result_shifted = torch.fft.fftshift(fft_result) 

frequencies_input_x = torch.fft.fftfreq(input.shape[0], 1.0 / (x1[1].item() - x1[0].item()))
frequencies_input_y = torch.fft.fftfreq(input.shape[1], 1.0 / (y1[1].item() - y1[0].item())) 
frequencies_x, frequencies_y = torch.meshgrid(frequencies_input_x ,frequencies_input_y, indexing='ij')

output=torch.from_numpy(output)
fft_result = torch.fft.fft2(output)
fft_result_shifted_out = torch.fft.fftshift(fft_result)  

fig = plt.figure(figsize=(15, 15))
ax1=fig.add_subplot(3,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), input.numpy(), cmap='viridis')
ax1.set_title('Original Input')
ax2 = fig.add_subplot(3,2,2)
ax2.imshow(torch.abs(fft_result_shifted).numpy(), extent=(frequencies_x.min(), frequencies_x.max(), frequencies_y.min(), frequencies_y.max()), origin='lower', cmap='viridis')
ax2.set_title('Intput Magnitude Spectrum')
ax2.set_xlabel('Frequency (u)')
ax2.set_ylabel('Frequency (v)')

ax3=fig.add_subplot(3,2,3, projection='3d')
ax3.plot_surface(x.numpy(), y.numpy(), output.numpy(), cmap='viridis')
ax3.set_title('Original Output')
ax4 = fig.add_subplot(3,2,4)
ax4.imshow(torch.abs(fft_result_shifted_out).numpy(), extent=(frequencies_x.min(), frequencies_x.max(), frequencies_y.min(), frequencies_y.max()), origin='lower', cmap='viridis')
ax4.set_title('Output Magnitude Spectrum')
ax4.set_xlabel('Frequency (u)')
ax4.set_ylabel('Frequency (v)')

#l=1
#x1 = torch.linspace(l/(s-1), D-l/(s-1), s-l*2)
#y1 = torch.linspace(l/(s-1), D-l/(s-1), s-l*2)
#Laplace_u=Laplace_u[0,l:-l,l:-l]

x1 = torch.linspace(0, D,s)
y1 = torch.linspace(0, D,s)
Laplace_u=Laplace_u[0,:,:]
x, y = torch.meshgrid(x1,y1, indexing='ij')
frequencies_input_x = torch.fft.fftfreq(input.shape[0], 1.0 / (x1[1].item() - x1[0].item()))
frequencies_input_y = torch.fft.fftfreq(input.shape[1], 1.0 / (y1[1].item() - y1[0].item())) 
frequencies_x, frequencies_y = torch.meshgrid(frequencies_input_x ,frequencies_input_y, indexing='ij')

fft_result = Laplace_hat(output,D).squeeze(0)
#fft_result1 = torch.fft.fft2(omega*input**2*output).squeeze(0)
#fft_result=fft_result-fft_result1

#fft_result = torch.fft.fft2(Laplace_u)
fft_result_shifted_Laplace_u = torch.fft.fftshift(fft_result).squeeze(0)

#fft_result = torch.fft.fft2(omega*input**2*output)
#fft_result_shifted_Laplace_u = torch.fft.fftshift(fft_result).squeeze(0)
#Laplace_u=(omega*input**2*output).squeeze(0)

ax5=fig.add_subplot(3,2,5, projection='3d')
ax5.plot_surface(x.numpy(), y.numpy(), Laplace_u.numpy(), cmap='viridis')
ax5.set_title('Laplace')
ax6 = fig.add_subplot(3,2,6)
ax6.imshow(torch.abs(fft_result_shifted_Laplace_u).numpy(), extent=(frequencies_x.min(), frequencies_x.max(), frequencies_y.min(), frequencies_y.max()), origin='lower', cmap='viridis')
ax6.set_title('Laplace Magnitude Spectrum')
ax6.set_xlabel('Frequency (u)')
ax6.set_ylabel('Frequency (v)')
plt.tight_layout()
plt.savefig('Frequency')


