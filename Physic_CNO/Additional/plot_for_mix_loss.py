import matplotlib.pyplot as plt
import os
from helper_functions.helper_for_plots import Find_text,plot_1
import numpy as np

Samplelist=[1,2,4,8,16,32,64,128]
Path_to_scratch='/cluster/scratch/harno'
folder_name=os.path.join(Path_to_scratch,'Best_Model_CNO_and_Physics-Informed_CNO')

error_list=[]
for samples in Samplelist:
    target_file='errors.txt'
    target_folder=f'Best_CNO_and_PI_CNO_{samples}'
    Path=os.path.join(folder_name,target_folder,target_file)
    best_error=Find_text(Path,"Best Testing Error: ")
    error_list.append(best_error)

plot_1(error_list,Samplelist,namefigsave='Plots/Error_of_Mix_loss',xname='Training Samples',title='$Loss=L_{Data}+\lambda L_{PDE}$')





