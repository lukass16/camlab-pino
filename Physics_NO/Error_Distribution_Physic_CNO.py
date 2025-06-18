import sys
sys.path.append("..")

import torch
import random
from random import randint
import numpy as np
from Error_Distribution import load_data, error_distribution, write_in_file_distribution,\
                               avg_spectra,write_in_file_spectra, plot_samples_different_models





#------------------------------------------------------------------------------------------------------------------------------------------------

# Set your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   "which" can be 

#   poisson             : Poisson equation 

which = "poisson"
in_dist = True

#Do you want to plot or to compute MEDIAN L1 errors? 
plot  = False 

if plot:
    device = "cpu"
    dist = False
else:
    dist = True

# Model folders
if which == "poisson":
    folder_CNO = "/cluster/scratch/harno/Best_Models_CNO/Best_CNO_poisson_1024"
    folder_Physic_CNO = "/cluster/scratch/harno/Best_Model_Physic_CNO/Best_CNO_poisson_0_1024"
    print(f'CNO folder: {folder_CNO}')
    print(f'Physics-Informed CNO folder: {folder_Physic_CNO}')

    N = 256


if dist:
    model_Physic_CNO = torch.load(folder_Physic_CNO + "/model.pkl", map_location=torch.device(device))
    modelCNO = torch.load(folder_CNO + "/model.pkl", map_location=torch.device(device))
    
    model_Physic_CNO.device = device
    modelCNO.device = device
    
    data_loader_CNO = load_data(folder_CNO, "CNO", device, which, in_dist = in_dist)

    
    data_loader_CNO.num_workers = 16

    
    E_Physic_CNO = error_distribution(which_model = "CNO", model = model_Physic_CNO, testing_loader = data_loader_CNO, p = 1, N = N, device = device, which = which)
    E_CNO = error_distribution(which_model = "CNO", model = modelCNO, testing_loader = data_loader_CNO, p = 1, N = N, device = device, which = which)
   
    if which == "airfoil":
        size = 128
    else:
        size = 64
    
    #Write distributions?
    write_dist = False
    if write_dist:
        dist_name = "dist.h5"
        write_in_file_distribution(folder_CNO, dist_name, E_CNO)
        write_in_file_distribution(folder_Physic_CNO, dist_name, E_Physic_CNO)
   
    
    #Write avg. spectra?
    write_spectra = False
    if write_spectra:
        spectra_name = "Spectra.h5"
        
        avg, avg_inp, avg_out = avg_spectra("CNO", modelCNO, data_loader_CNO, device, in_size = size)
        write_in_file_spectra(folder_CNO, spectra_name, avg, avg_inp, avg_out)
           
        avg, avg_inp, avg_out = avg_spectra("Physic_CNO", model_Physic_CNO, data_loader_CNO, device, in_size = size)
        write_in_file_spectra(folder_Physic_CNO,spectra_name, avg, avg_inp, avg_out)
        
        dist_name = "dist.h5"
        write_in_file_distribution(folder_CNO, dist_name, E_CNO)
        write_in_file_distribution(folder_Physic_CNO, dist_name, E_Physic_CNO)

    
    print("-------------")
    print("Experiment: ", which)
    print("in_dist = " + str(in_dist))
    print("")
    print("CNO error:", np.median(E_CNO))
    print("Physics-Informed CNO error:", np.median(E_Physic_CNO))
    print("-------------")


elif plot:
        
    model_Physic_CNO = torch.load(folder_Physic_CNO + "/model.pkl", map_location=torch.device(device))
    modelCNO = torch.load(folder_CNO + "/model.pkl", map_location=torch.device(device))

    
    model_Physic_CNO.device = device
    modelCNO.device = device
    
    in_dist = False
    
    data_loader_CNO = load_data(folder_CNO, "CNO", device, which, batch_size = 1, in_dist = in_dist)
    
    random.seed()
    n = randint(0,N)

    data_loader_CNO.num_workers = 0

    random.seed()
    n = randint(0,N)
    plot_samples_different_models(["CNO", "Physics-Informed CNO"], 
                                  [data_loader_CNO, data_loader_CNO], 
                                  [modelCNO, model_Physic_CNO], 
                                  1, n, which = which)