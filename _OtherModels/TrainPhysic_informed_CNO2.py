import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Problems.Benchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
from helper_functions.fine_tuning_helper import  freezing_parameters
from loss_functions.Pde_loss import loss_pde
if len(sys.argv) == 1:

    training_properties = {
        "learning_rate": 0.0005, 
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 500,
        "batch_size": 16,
        "exp": 1, #Do we use L1 or L2 errors? Default: L1
        "training_samples": 256, #How many training samples?
        "pde_decay": 0.5 
    }

    model_architecture_ = {
       
        #----------------------------------------------------------------------
        #Parameters to be chosen with model selection:
            
        "N_layers": 4, #Number of (D) + (U) layers. In our experiments, N_layers must be even.
        "kernel_size": 3, #Kernel size.
        "channel_multiplier": 32, #Parameter d_e (how the number of channels changes)
        
        "N_res": 8, #Number of (R) blocks.
        "res_len": 2, #Coefficienr r in (R) definition.
        
        #----------------------------------------------------------------------
        #Parameters that depend on the problem: 
        
        "in_size": 64, #Resolution of the computational grid
        "retrain": 4, #Random seed
        
        #----------------------------------------------------------------------
        #We fix the following parameters:
        
        #Filter properties:
        "cutoff_den": 2.0001, #
        "lrelu_upsampling": 2, #Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 1, #Coefficient c_h. Default is 1
        "filter_size": 6, # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0, #Is the filter radially symmetric? Default is 0 - NO.

        "FourierF": 0, #Number of Fourier Features in the input channels. Default is 0.

        #----------------------------------------------------------------------
    }

    fine_tuning= {
        #----------------------------------------------------------------------
        #Load Trained model: (Must be compatible with model_architecture)

        "Load pretrained model": True,
        "Path to pretrained model": '/cluster/scratch/harno/SELECTED_MODELS/',
        "Freezing encoder or decoder": "Decoder",
        "Percentage to freeze": 0.5   #If zero not part is freezed and the complet network is trained
        #----------------------------------------------------------------------
    }
    
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    

    which_example = "poisson"

    # Save the models here:
    folder = "TrainedModels/"+"Physic_informed_CNO"+which_example
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
lampda=training_properties['pde_decay']



if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')


if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
elif which_example == "wave_0_5":
    example = WaveEquation(model_architecture_, device, batch_size, training_samples)
elif which_example == "allen":
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
elif which_example == "cont_tran":
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
else:
    raise ValueError()



#-----------------------------------Load pretrained model--------------------------------------

model = example.model

if fine_tuning["Load pretrained model"]: 
     df = pd.DataFrame.from_dict([fine_tuning]).T
     df.to_csv(folder + '/fine_tuning.txt', header=False, index=True, mode='w')
 
     pretrained_model_path = fine_tuning["Path to pretrained model"]+'\model.pkl'
     net_architucture_path = fine_tuning["Path to pretrained model"]+'\\net_architecture.txt'
     freezing=fine_tuning["Percentage to freeze"]
     encoder_or_decoder=fine_tuning["Freezing encoder or decoder"]

     if not os.path.exists(pretrained_model_path):
         raise FileNotFoundError(f"The model file '{pretrained_model_path}' does not exist.")
     
     #model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
     model = torch.load(pretrained_model_path, map_location=device)
     df = pd.read_csv(net_architucture_path, header=None, index_col=0)
     model_architecture = df.to_dict()[1]

     print(f'Loading trained netrwork from {pretrained_model_path}')

     if freezing!=0:
          freezing_parameters(model,model_architecture_,encoder_or_decoder=encoder_or_decoder,
                              freezing=freezing)
     else:
          print(f'Complete network is trained')
else:
     print(f'Netrwork trained from scratch')

#-----------------------------------Train--------------------------------------------

n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

#In this case we try to train with the same data!!
#Later the idea is to create new data, but I am not sure from which distribution I should draw the data.

if p == 1:
    loss = torch.nn.L1Loss()

elif p == 2:
    loss = torch.nn.MSELoss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.1 * epochs)    # Earlt stopping parameter
counter = 0

for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        
        #Disable : Should we disable the printing of the error report per epoch?
        
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        train_f=0.0
        train_pde=0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            output_pred_batch = model(input_batch)
            
            
            if which_example == "airfoil": #Mask the airfoil shape
                output_pred_batch[input_batch==1] = 1
                output_batch[input_batch==1] = 1

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)
            loss_PDE= loss_pde(output_pred_batch,input_batch,which_example,p=p)
            loss_total=loss_f+loss_PDE*lampda
            loss_total.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_total.item() / (step + 1)
            train_f = train_f * step / (step + 1) + loss_f.item() / (step + 1)
            train_pde = train_pde * step / (step + 1) + loss_PDE.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse,'Data loss': train_f,'PDE loss': train_pde})

        writer.add_scalar("train_loss/train_loss", train_mse, epoch)
        
        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            for step, (input_batch, output_batch) in enumerate(val_loader):
                
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                if which_example == "airfoil": #Mask the airfoil shape
                    output_pred_batch[input_batch==1] = 1
                    output_batch[input_batch==1] = 1
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    if which_example == "airfoil": #Mask the airfoil shape
                        output_pred_batch[input_batch==1] = 1
                        output_batch[input_batch==1] = 1

                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)

            writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter+=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()

    if counter>patience:
        print("Early Stopping")
        break
