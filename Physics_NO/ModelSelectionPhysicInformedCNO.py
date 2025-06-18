import itertools
import os
import sys
import json
import numpy as np

np.random.seed(0)

which_example = "helmholtz"
Pretrainedsampels=5008
folder_name = "/cluster/scratch/harno" + f"/MODEL_SELECTION_PhysicCNO_test_fake_{which_example}_"+str(Pretrainedsampels)
cluster = "true"

Path_to_Pretrained_Network='/cluster/home/harno/Code/PrivateConvolutionalNeuralOperator/Physic_CNO/scratch/Best_CNO_Helmholtz/Best_CNO_Helmholtz_pad_10'

#Path_to_Pretrained_Network='None'

InfoPretrainedNetwork= {
        "Path to pretrained model": Path_to_Pretrained_Network, 
        "Pretrained Samples": Pretrainedsampels   
        }

all_training_properties = {
    "learning_rate": [0.001],#0.0003
    "weight_decay": [1e-10],
    "scheduler_step": [10],
    "scheduler_gamma": [0.98],
    "epochs": [600],
    "batch_size": [32],
    "exp": [3],#1
    "training_samples": [5008],
    "pde_decay": [8],#4, 20
    "boundary_decay":[8],#8
    "pad_factor": [10]
}


#Only relevant if Path to pretrained model==None
all_model_architecture = {
    "in_size": [128+10],
    "N_layers": [4],
    "N_res": [8],
    "N_res_neck": [8],
    "kernel_size": [3],
    "channel_multiplier": [16],
    "retrain": [7],#4 
    "half_width_mult": [1],
    "FourierF": [0]
}

ndic = {**all_training_properties,
        **all_model_architecture}

print(folder_name)
old_folders=0
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
else:
    old_folders=len(os.listdir(folder_name))
    
settings = list(itertools.product(*ndic.values()))

sbatch = True

i = 0+old_folders
for setup in settings:
    # time.sleep(10)
    print(setup)

    folder_path = "\'" + folder_name +"/"+str(setup[7])+"Setup_" + str(i) + "\'"
    
    #folder_path = "$SCRATCH"+"\'" + folder_name + "/Setup_" + str(i) + "\'"

    print(folder_path)
    print("###################################")
    training_properties_ = {
    }
    j = 0
    for k, key in enumerate(all_training_properties.keys()):
        training_properties_[key] = setup[j]
        j = j + 1

    model_architecture_ = {
    }
    for k, key in enumerate(all_model_architecture.keys()):
        model_architecture_[key] = setup[j]
        j = j + 1

    arguments = list()
    arguments.append(folder_path)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if sbatch:
            arguments.append("\\\"" + str(training_properties_) + "\\\"")
        else:
            arguments.append("\'" + str(training_properties_).replace("\'", "\"") + "\'")

    else:
        arguments.append(str(training_properties_).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if sbatch:
            arguments.append("\\\"" + str(model_architecture_) + "\\\"")
        else:
            arguments.append("\'" + str(model_architecture_).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(model_architecture_).replace("\'", "\""))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if sbatch:
            arguments.append("\\\"" + str(InfoPretrainedNetwork) + "\\\"")
        else:
            arguments.append("\'" + str(InfoPretrainedNetwork).replace("\'", "\"") + "\'")

    else:
        arguments.append(str(InfoPretrainedNetwork).replace("\'", "\""))

    arguments.append(which_example)
    print(arguments)

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            if sbatch:
                string_to_exec = "sbatch --output=Out/PCNO%j.out --time=4:00:00 -n 16 -G 1 --mem-per-cpu=512 --wrap=\"python3 TrainPhysic_informed_CNO_Module.py"
            else:
                string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 TrainPhysic_informed_CNO_Module.py"
        else:
            string_to_exec = "python3 TrainPhysic_informed_CNO_Module.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        if cluster and sbatch:
            string_to_exec = string_to_exec + " \""
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
