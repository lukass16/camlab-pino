import copy
import json
import os
import sys
sys.path.append("..")

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from Problems.CNOBenchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer, Darcy, Helmholtz
from loss_functions.Pde_loss import loss_pde,Operator_loss
from loss_functions.Relative_loss import Relative_loss


if len(sys.argv) == 2:

    InfoPretrainedNetwork= {
        #----------------------------------------------------------------------
        #Load Trained model: (Must be compatible with model_architecture)
        #Path to pretrained model: None for training from scratch
        "Path to pretrained model": None, 
        "Pretrained Samples":  5008,
    }
    training_properties = {
        "learning_rate": 0.0003, 
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 600,
        "batch_size": 16,
        "exp": 1,                # Do we use L1 or L2 errors? Default: L1
        "training_samples": 5008,  # How many training samples?
        "pde_decay": 4,
        "boundary_decay":8,
        "pad_factor": 10 #0 if you dont want to pad the input
    }
 #---------- Add model Parameters for training form scratch----------------------
 #model_architecture_ is only relevant if Path to pretrained model==None!!
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "N_layers": 4,            # Number of (D) & (U) blocks 
        "channel_multiplier": 16, # Parameter d_e (how the number of channels changes)
        "N_res": 5,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 5,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_size": 128+10*2,            # Resolution of the computational grid
        "retrain": 4,             # Random seed
        "kernel_size": 3,         # Kernel size.
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "activation": 'cno_lrelu',# cno_lrelu or lrelu
        
        #Filter properties:
        "cutoff_den": 2.0001,     # Cutoff parameter.
        "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
        "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    }
   
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   helmholtz           : Helmholtz equation
    
    which_example = sys.argv[1]

    # Save the models here:
    folder = "/cluster/scratch/harno/TrainedModels/"+"Physic_informed_CNO2_pad"+which_example
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    InfoPretrainedNetwork = json.loads(sys.argv[4].replace("\'", "\""))
    if InfoPretrainedNetwork["Path to pretrained model"]=='None':
       InfoPretrainedNetwork["Path to pretrained model"]=None
    which_example = sys.argv[5]


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
Boundary_decay=training_properties['boundary_decay']
pad_factor=training_properties['pad_factor']



if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
if  InfoPretrainedNetwork["Path to pretrained model"] is not None: 
     
     df = pd.DataFrame.from_dict([InfoPretrainedNetwork]).T
     df.to_csv(folder + '/InfoPretrainedNetwork.txt', header=False, index=True, mode='w')
 
     pretrained_model_path = InfoPretrainedNetwork["Path to pretrained model"]+'/model.pkl'
     net_architucture_path = InfoPretrainedNetwork["Path to pretrained model"]+'/net_architecture.txt'
     df = pd.read_csv(net_architucture_path, header=None, index_col=0)
     model_architecture_ = df.to_dict()[1]
     model_architecture_ = {key: int(value) if str(value).isdigit() else float(value) if '.' in str(value) else value for key, value in df.to_dict()[1].items()}
     df = pd.DataFrame.from_dict([model_architecture_]).T
     df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
elif which_example == "helmholtz":
    example = Helmholtz(model_architecture_, device, batch_size, training_samples,s=128,N_max=19675,pad_factor=pad_factor)
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

if  InfoPretrainedNetwork["Path to pretrained model"] is not None: 

     if not os.path.exists(pretrained_model_path):
         raise FileNotFoundError(f"The model file '{pretrained_model_path}' does not exist.")
     
     model = torch.load(pretrained_model_path, map_location=device)
     model_fix=torch.load(pretrained_model_path, map_location=device)


     for param in model_fix.parameters():
        param.requires_grad = False

     print(f'Loading trained network from {pretrained_model_path}')

else:
     print(f'Network is trained from scratch')
     df = pd.DataFrame.from_dict([model_architecture_]).T
     df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

#-----------------------------------Train--------------------------------------------

n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

Normalization_values=train_loader.dataset.get_max_and_min() #Get max_min of the data

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")


best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)    # Early stopping parameter
counter = 0
history_trainOP=list()
history_trainPDE=list()
history_train=list()
history_val=list()

loss_relative=Relative_loss(pad_factor,model_architecture_["in_size"])

for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        train_op=0.0
        loss_OP=0.0
        train_pde=0.0
        loss_PDEL1=0
        loss_OPL1=0
        running_relative_train_mse = 0.0
        for step, (input_batch,_) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            

            output_pred_batch = model(input_batch)
            
            if which_example == "airfoil": #Mask the airfoil shape
                output_pred_batch[input_batch==1] = 1
                   
            #Important for the pde_loss to work input and output should be normalized!
            loss_PDE= loss_pde(output_pred_batch,input_batch,which_example,Normalization_values,p=p,Boundary_decay=Boundary_decay)
            if  InfoPretrainedNetwork["Path to pretrained model"] is not None:
                 with torch.no_grad():
                     output_fix=model_fix(input_batch)
                 loss_op=Operator_loss(output_train=output_pred_batch,output_fix=output_fix,p=p)
                 loss_total=loss_op*lampda+loss_PDE
                 loss_OP=loss_op.item()
                 
            else:
                 loss_total=loss_PDE

            loss_PDEL1+=loss_PDE.item() #values for plot
            loss_OPL1+=loss_OP #values for plot
            loss_total.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_total.item() / (step + 1)
            train_op = train_op * step / (step + 1) + loss_OP/ (step + 1)
            train_pde = train_pde * step / (step + 1) + loss_PDE.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse,'Operator loss': train_op,'PDE loss': train_pde})
        
        history_trainOP.append(loss_OPL1/len(train_loader))
        history_trainPDE.append(loss_PDEL1/len(train_loader))
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
                
                loss_f =loss_relative(output_pred_batch,output_batch)
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)
            history_val.append(test_relative_l2)
            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    if which_example == "airfoil": #Mask the airfoil shape
                        output_pred_batch[input_batch==1] = 1
                        output_batch[input_batch==1] = 1

                    loss_f =loss_relative(output_pred_batch,output_batch)
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)
            history_train.append(train_relative_l2)
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].grid(True, which="both", ls=":")
    axes[0].plot(torch.arange(start=0, end=len(history_trainPDE)).numpy(), history_trainPDE, label='Loss PDE')
    axes[0].plot(torch.arange(start=0, end=len(history_trainOP)).numpy(), history_trainOP, label='Loss OP')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].grid(True, which="both", ls=":")
    axes[1].plot(torch.arange(start=0, end=len(history_train)).numpy(), history_train, label='Training loss')
    axes[1].plot(torch.arange(start=0, end=len(history_val)).numpy(), history_val, label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(folder+'/Losses.png')
    plt.close(fig)
    
print('Finish')

