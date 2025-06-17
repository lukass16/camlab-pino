import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_losses(path_to_file):
    reader=h5py.File(path_to_file,'r')
    #training_loss=np.array(reader["Train error"])
    Boundary_decay=1
    validation_loss=np.array(reader["Validation error"])
    pde_loss=np.array(reader["PDE error"])
    boundary_loss=np.array(reader["Boundary error"])
    training_loss=pde_loss+Boundary_decay*boundary_loss
    return [training_loss, validation_loss]

def plot_losses(Losses,name):
    fig,ax=plt.subplots(1,2,figsize=(8,5))
    for key in Losses.keys():
        ax[0].plot(np.arange(len(Losses[key][0])),Losses[key][0],label=key)
        ax[1].plot(np.arange(len(Losses[key][1])),Losses[key][1],label=key)
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Relative Loss')
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[0].set_title('Training loss')
    ax[1].set_title('Validation loss')
    plt.tight_layout()
    plt.savefig(name)

Path_to_L1 ='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/Best_Model_Physic_CNO/Best_CNO_poisson_0_128/Training_error.h5'
Path_to_L2 ='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_No_Pretraining_poisson/128Setup_11/Training_error.h5'
Path_to_PL1 ='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_No_Pretraining_poisson/128Setup_9/Training_error.h5'
Path_to_PL2='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/MODEL_SELECTION_PhysicCNO_No_Pretraining_poisson/128Setup_10/Training_error.h5'
Losses={}
Losses['Model']= load_losses(Path_to_L1)
Losses['Preconditioned Model']= load_losses(Path_to_PL1)


plot_losses(Losses,'Preconditioning Loss L1')

Losses={}
Losses['Model']= load_losses(Path_to_L2)
Losses['Preconditioned Model']=load_losses(Path_to_PL2)

plot_losses(Losses,'Preconditioning Loss L2')

Losses={}
Losses['Model L1']= load_losses(Path_to_L1)
Losses['Model L2']= load_losses(Path_to_L2)
Losses['Preconditioned Model L1']= load_losses(Path_to_PL1)
Losses['Preconditioned Model L2']= load_losses(Path_to_PL2)

plot_losses(Losses,'Preconditioning Loss')



