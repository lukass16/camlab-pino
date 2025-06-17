import copy
import json
import os
import sys
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from Problems.CNOBenchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer, Darcy, Helmholtz
from Physic_CNO.loss_functions.Relative_loss import Relative_loss,Relative_error_training


"""-------------------------------Setting parameters for training--------------------------------"""
if len(sys.argv) == 2:
    
    training_properties = {
        "learning_rate": 0.0007, 
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 300,
        "batch_size": 16,
        "exp": 1,                # Do we use L1 or L2 errors? Default: L1
        "training_samples": 1024,    # How many training samples?
        "pad_factor": 0         # No padding needed for 64x64 data
    }
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "N_layers": 4,             # Number of (D) & (U) blocks 
        "channel_multiplier": 16,  # Parameter d_e (how the number of channels changes)
        "N_res": 5,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 5,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_size": 64,             # Resolution of the computational grid (matches data)
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
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    #   darcy               : Darcy Flow

    which_example = sys.argv[1]
    #which_example = "poisson"
    #which_example = "shear_layer"

    # Save the models here:
    training_samples=training_properties["training_samples"]
    folder = "/cluster/home/lkellijs/camlab-pino/TrainedModels/"+"CNO_"+str(training_samples)+which_example #! Change this to your own path
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using the device: {device}')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
pad_factor=training_properties["pad_factor"]

#net_architucture_path='/cluster/scratch/harno/SELECTED_MODELS/Best_poisson_CNO/net_architecture.txt'
#df = pd.read_csv(net_architucture_path, header=None, index_col=0)
#model_architecture = df.to_dict()[1]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

# save the training properties and the model architecture
df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')


"""------------------------------------Load the data--------------------------------------"""
if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples, size = 64)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples) # automatically uses s=64
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
    model_architecture_["in_size"] = 128
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "darcy":
    example = Darcy(model_architecture_, device, batch_size, training_samples)
else:
    raise ValueError()
    
"""------------------------------------Train--------------------------------------"""
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)    # Early stopping parameter
counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")


history_train=list()
history_val=list()
history_trainL1=list()
in_size=model_architecture_["in_size"]
loss_relative=Relative_loss(pad_factor,in_size)
loss_train=Relative_error_training(p=p,pad_factor=pad_factor,input_size=in_size,device=device)

for epoch in range(epochs):
    
    # each epoch
    with tqdm(unit="batch", disable=False) as tepoch:
        
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        
        # each batch in the training loader
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            output_pred_batch = model(input_batch)
            
            if which_example == "airfoil": #Mask the airfoil shape
                output_pred_batch[input_batch==1] = 1
                output_batch[input_batch==1] = 1

            loss_f =loss_train(output_pred=output_pred_batch,output=output_batch)
            
            #if which_example=="helmholtz":
            #    #Add more importance to boundary
            #    factor=20
            #    loss_x0=loss(output_pred_batch[:,:,0,:], output_batch[:,:,0,:]) / loss(torch.zeros_like(output_batch[:,:,0,:]).to(device), output_batch[:,:,0,:])
            #    loss_xD=loss(output_pred_batch[:,:,-1,:], output_batch[:,:,-1,:]) / loss(torch.zeros_like(output_batch[:,:,-1,:]).to(device), output_batch[:,:,-1,:])
            #    loss_y0=loss(output_pred_batch[:,:,:,0], output_batch[:,:,:,0]) / loss(torch.zeros_like(output_batch[:,:,:,0]).to(device), output_batch[:,:,:,0])
            #    loss_yD=loss(output_pred_batch[:,:,:,-1], output_batch[:,:,:,-1]) / loss(torch.zeros_like(output_batch[:,:,:,-1]).to(device), output_batch[:,:,:,-1])
            #    loss_b=(loss_x0+loss_xD+loss_y0+loss_yD)*0.25
            #    loss_f=loss_f+loss_b*factor
            
            history_trainL1.append(loss_f.item())
            loss_f.backward() 
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

        writer.add_scalar("train_loss/train_loss", train_mse, epoch)
        
        # after each epoch, we evaluate the model on the validation a set
        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            # each batch in the validation loader
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

            # each batch in the training loader
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
                torch.save(best_model, folder + "/model.pkl") #! save the best model
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter+=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        # Save most recent errors to a file
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()

    # Early stopping
    if counter>patience:
        print("Early Stopping")
        break

    # Plot losses
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[0].set_ylim(0,max(history_train))
    axes[0].grid(True, which="both", ls=":")
    axes[0].plot(torch.arange(start=0, end=len(history_train)).numpy(), history_train, label='Training loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].grid(True, which="both", ls=":")
    axes[1].plot(torch.arange(start=0, end=len(history_val)).numpy(), history_val, label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[2].grid(True, which="both", ls=":")
    axes[2].plot(torch.arange(start=0, end=len(history_trainL1)).numpy(),history_trainL1,label='Training L1')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Loss')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(folder+'/Losses.png')
    plt.close(fig)
    
print("Finished Training")