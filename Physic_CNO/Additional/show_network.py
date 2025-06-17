import io
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
sys.path.append('../..')
import torch
import h5py
import numpy as np
from loss_functions.Relative_loss import Relative_loss
from loss_functions.ModulePDELoss import Loss_PDE,Laplace

#Show Bogdan for Bad example
path_to_trained_model='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_NewModule_helmholtz_5008/5008Setup_2/model.pkl'

#Good example but to high decay
#path_to_trained_model='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_NewModule_helmholtz_5008/5008Setup_1/model.pkl'

#Show for oscillation
#path_to_trained_model='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/Best_CNO_Helmholtz/Best_CNO_Helmholtz_no_pad/model.pkl'

#No padding but with physic
#path_to_trained_model='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/TrainedModels/Physic_informed_CNO_helmholtz/model.pkl'

#load images
file_to_data='/cluster/scratch/harno/data/HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5'
n=np.random.randint(200)
reader = h5py.File(file_to_data, 'r')
input= torch.from_numpy(reader['Sample_' + str(n)]["a"][:]).to(torch.float32).unsqueeze(0)
output=torch.from_numpy(reader['Sample_' + str(n)]["u"][:]).to(torch.float32).unsqueeze(0)
boundary=torch.from_numpy(np.array(reader['Sample_' + str(n)]["bc"])).to(torch.float32)
shape=input.shape
pad_factor=10
old_shape=shape[-1]
input_pad=torch.zeros((*shape[0:-2],shape[-2]+pad_factor,shape[-1]+pad_factor))
input_pad[...,:old_shape,:old_shape]=input-1
input=input_pad

if pad_factor==0:
   bc    =  torch.zeros_like(input)
   bc[...,0,:]  = boundary
   bc[...,-1,:] = boundary
   bc[...,:,0]  = boundary
   bc[...,:,-1] = boundary
else:
   bc    =  torch.ones_like(input)*boundary

input_net = torch.cat((input_pad, bc), 0)




#load model
model = torch.load(path_to_trained_model, map_location=input.device)
pred=model(input_net.unsqueeze(0)).squeeze(0)
mean=0.11523915668552
std = 0.8279975746000605
pred_un = pred*std+mean
error=abs(pred_un[...,:old_shape,:old_shape]-output)
Laplace_f=Laplace(s=old_shape+pad_factor,D=1+pad_factor/old_shape)

Normalization_values= {
        "mean_data": 1,
        "mean_model":mean ,
        "std_model": std
         }

loss=Relative_loss(input_size=old_shape+pad_factor,pad_factor=pad_factor)
loss_pde=Loss_PDE(which_example='helmholtz',Normalization_values=Normalization_values,p=1,\
                  Boundary_decay=1,pad_factor=pad_factor,in_size=old_shape+pad_factor,D=1)
print('Loss:' ,     loss(pred,(output-mean)/std).item())
print('Loss PDE:' , loss_pde(input_net.unsqueeze(0),pred).item())

#Create grid points
D, s=1,(128+pad_factor)
x, y = torch.meshgrid(torch.linspace(0, D, s), torch.linspace(0, D, s) ,indexing='ij')
#Plot input and output
fig=plt.figure()
ax1=fig.add_subplot(2,2,1, projection='3d')
ax1.plot_surface(x.numpy(), y.numpy(), input[0,...].numpy(), cmap='viridis')
ax1.set_title('a')
ax2=fig.add_subplot(2,2,2, projection='3d')
ax2.plot_surface(x[:old_shape,:old_shape].numpy(), y[:old_shape,:old_shape].numpy(), output[0,...].numpy(), cmap='viridis')
ax2.set_title('u')
ax3=fig.add_subplot(2,2,3, projection='3d')
ax3.plot_surface(x.numpy(), y.numpy(), pred_un[0,...].detach().numpy(), cmap='viridis')
ax3.set_title('u_pred')
ax4=fig.add_subplot(2,2,4, projection='3d')
ax4.plot_surface(x[:old_shape,:old_shape].numpy(), y[:old_shape,:old_shape].numpy(), error[0,...,:old_shape,:old_shape].detach().numpy(), cmap='viridis')
ax4.set_title('error')
plt.tight_layout()
plt.savefig('PredModel')



fig,ax=plt.subplots(2,2)
ax[0,0].imshow(input[0,...].numpy(), cmap="gist_ncar")
ax[0,0].title.set_text('a')

ax[0,1].imshow(output[0,...].numpy(), cmap="gist_ncar")
ax[0,1].title.set_text('u')

ax[1,0].imshow(pred_un[0,...].detach().numpy(), cmap="gist_ncar")
ax[1,0].title.set_text('u_pred')

ax[1,1].imshow(error[0,...].detach().numpy(), cmap="gist_ncar")
ax[1,1].title.set_text('error')
plt.show()
plt.tight_layout()
plt.savefig('PredModel_im')


pred_un=pred_un.squeeze(1)
Laplace_pred=Laplace_f(pred_un)
input_un=input[...,:old_shape,:old_shape]+1
AsquardU= input_un**2*output[...,:old_shape,:old_shape]*(5*torch.pi/2)**2
loss_f = torch.nn.L1Loss()
loss=loss_f(Laplace_pred[...,:old_shape,:old_shape],\
                            -(5*torch.pi/2)**2*input_un**2*pred_un[...,:old_shape,:old_shape])

loss1=loss_f(Laplace_pred[...,:old_shape,:old_shape],\
                            -(5*torch.pi/2)**2*input_un**2*output[...,:old_shape,:old_shape])

boundary=boundary.unsqueeze(-1)
boundary_lossx_0=loss_f(pred_un[...,0,:old_shape],  torch.mul(boundary,torch.ones_like(output[...,0,:old_shape])))
boundary_lossy_0=loss_f(pred_un[...,:old_shape,0],  torch.mul(boundary,torch.ones_like(output[...,:old_shape,0])))
boundary_lossx_D=loss_f(pred_un[...,-1-pad_factor,:old_shape], torch.mul(boundary,torch.ones_like(output[...,-1-pad_factor,:old_shape])))
boundary_lossy_D=loss_f(pred_un[...,:old_shape,-1-pad_factor], torch.mul(boundary,torch.ones_like(output[...,:old_shape,-1-pad_factor])))
boundary_loss=0.25*(boundary_lossx_0+boundary_lossy_0+boundary_lossx_D+boundary_lossy_D)
print('PDE Loss:' ,loss.item())
print('PDE Loss with correct output:' ,loss1.item())
print('Boundary loss:' ,boundary_loss.item())

fig,ax=plt.subplots(1,2)
im1=ax[0].imshow(-Laplace_pred[0,...,:old_shape,:old_shape].detach().numpy().T, cmap="gist_ncar")
ax[0].set_title('-Laplace_u')

im2=ax[1].imshow(AsquardU[0,...].detach().numpy().T, cmap="gist_ncar")
ax[1].set_title('a**2*u*omega')
fig.colorbar(im1,ax=ax[0])
fig.colorbar(im2,ax=ax[1])
plt.show()
plt.tight_layout()
plt.savefig('PredModel_Laplace')

