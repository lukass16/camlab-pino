import h5py
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
sys.path.append('../..')
import torch
import h5py
import numpy as np
from loss_functions.Relative_loss import Relative_loss
from loss_functions.ModulePDELoss import Loss_PDE,Laplace
from helper_functions.Preconditioning import Create_P,Precondition_output


path_to_network='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/Best_Model_Physic_CNO/Best_CNO_poisson_0_128/model.pkl'
#path_to_network='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_No_Pretraining_poisson/128Setup_6/model.pkl'
#path_to_network='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_No_Pretraining_poisson/128Setup_9/model.pkl'
which_example='poisson'
Preconditioning=False
in_size=64
D=2

if Preconditioning:
    P=Create_P(which_example,in_size,D=D,device='cpu')

file_to_data='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/data/PoissonData_PDEDomain2.h5'
n=np.random.randint(200)
reader = h5py.File(file_to_data, 'r')
input= torch.from_numpy(reader['Sample_' + str(n)]["input"][:]).to(torch.float32).unsqueeze(0)
output=torch.from_numpy(reader['Sample_' + str(n)]["output"][:]).to(torch.float32).unsqueeze(0)
min_data  = torch.from_numpy(np.array(reader['min_inp']))
max_data  = torch.from_numpy(np.array(reader['max_inp']))
min_model = torch.from_numpy(np.array(reader['min_out']))
max_model = torch.from_numpy(np.array(reader['max_out']))
input_norm=(input-min_data)/(max_data-min_data)

model = torch.load(path_to_network, map_location=input.device)
pred_norm=model(input_norm.unsqueeze(0)).squeeze(0)
pred=pred_norm*(max_model-min_model)+min_model
if Preconditioning:
    pred=Precondition_output(pred,P)
    pred_norm=(pred-min_model)/(max_model-min_model)
output_norm=(output-min_model)/(max_model-min_model)

loss=Relative_loss(input_size=in_size,pad_factor=0)

loss1=loss(pred,output)
loss2=loss(pred_norm,output_norm)

print(loss1.item(),loss2.item())




