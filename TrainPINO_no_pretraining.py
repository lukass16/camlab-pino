import copy
import json
import os
import sys
import h5py

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from Problems.FNOBenchmarks import SinFrequency
from Physic_CNO.loss_functions.Relative_loss import Relative_loss
from Physic_CNO.loss_functions.ModulePDELoss import Loss_PDE
from Physic_CNO.helper_functions.Preconditioning import Unnormalize_for_testing, Normalize_for_testing, Precondition_output, Create_P


if len(sys.argv) == 2:

    training_properties = {
       "learning_rate": 0.0003, 
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 100,
        "batch_size": 16,
        "exp": 3,                # Do we use L1 or L2 errors? Default: L1 3 for smooth
        "training_samples": 128,  # How many training samples?
        "boundary_decay":1,
        "pad_factor": 0,
        "preconditoning": False
    }

    fno_architecture_ = {
        "width": 64,
        "modes": 16,
        "FourierF" : 0, #Number of Fourier Features in the input channels. Default is 0.
        "n_layers": 4, #Number of Fourier layers
        "padding": 0,
        "include_grid":1,
        "retrain": 4, #Random seed
    }

   
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   helmholtz           : Helmholtz equation
    
    which_example = sys.argv[1]

    # Save the models here:
    folder = "/cluster/scratch/harno/TrainedModels/"+"PINO_no_pretraining"+which_example
        
else:
    
    # Do we use a script to run the code (for cluster):
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    if training_properties['preconditioning']=='true':
        training_properties['preconditioning']=True
    else: 
        training_properties['preconditioning']=False
    fno_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
writer = SummaryWriter(log_dir=folder)

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
preconditioning=training_properties['precondtioning']
in_size=fno_architecture_["width"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

if preconditioning:
    print('Load P')
    P=Create_P(which_example=which_example,s=in_size,device=device,D=2)

if which_example == "poisson":
    example = SinFrequency(fno_architecture_, device, batch_size,training_samples)
else:
    raise ValueError("the variable which_example has to be one between darcy")


df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([fno_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')


model = example.model


#-----------------------------------Train--------------------------------------------

n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

Normalization_values=train_loader.dataset.get_max_and_min() #Get max_min of the data

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

patience = int(0.25 * epochs)
best_model_testing_error = 300

loss_relative=Relative_loss(pad_factor,in_size)
loss_pde=Loss_PDE(which_example=which_example,Normalization_values=Normalization_values,p=p,\
                  pad_factor=pad_factor,in_size=in_size,preconditioning=preconditioning,device=device)
unnormalize=Unnormalize_for_testing(which_example,Normalization_values)
normalize=Normalize_for_testing(which_example,Normalization_values)

counter = 0
losses = {'loss_PDE': [], 'loss_boundary': [], 'loss total': [], 'loss_training': [], 'loss_validation':[]}
for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        
        losses['loss_PDE'].append(0)
        losses['loss_boundary'].append(0)
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

            loss_PDE, loss_boundary= loss_pde(input=input_batch.view(batch_size,-1,in_size,in_size),output=output_pred_batch.view(batch_size,-1,in_size,in_size))

            loss_total=loss_PDE+Boundary_decay*loss_boundary

            losses['loss_PDE'][-1]     +=loss_PDE.item()      #values for plot
            losses['loss_boundary'][-1]+=loss_boundary.item() #values for plot
            loss_total.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_total.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

        losses['loss_boundary'][-1]/=len(train_loader)
        losses['loss_PDE'][-1]/=len(train_loader)          
        writer.add_scalar("train_loss/train_loss", train_mse, epoch)

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0

            for step, (input_batch, output_batch) in enumerate(val_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    if preconditioning:
                           output_pred_batch=unnormalize(output_pred_batch)
                           output_pred_batch=Precondition_output(output_pred_batch,P)
                           output_pred_batch=normalize(output_pred_batch)
                    
                    loss_f=loss_relative(output_pred_batch.view(batch_size,-1,in_size,in_size),output_batch.view(batch_size,-1,in_size,in_size))
                    test_relative_l2 += loss_f.item()

            test_relative_l2 /= len(val_loader)
            losses['loss_validation'].append(test_relative_l2)

            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    if preconditioning:
                         output_pred_batch=unnormalize(output_pred_batch)
                         output_pred_batch=Precondition_output(output_pred_batch,P)
                         output_pred_batch=normalize(output_pred_batch)
                    
                    loss_f = loss_relative(output_pred_batch.view(batch_size,-1,in_size,in_size),output_batch.view(batch_size,-1,in_size,in_size))
                    train_relative_l2 += loss_f.item()

            train_relative_l2 /= len(train_loader)
            losses['loss_training'].append(test_relative_l2)
            
            writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                counter = 0
            else:
                counter +=1

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
    axes[0].plot(torch.arange(start=0, end=len(losses['loss_PDE'])).numpy(), losses['loss_PDE'], label='Loss PDE')
    axes[0].plot(torch.arange(start=0, end=len(losses['loss_boundary'])).numpy(), losses['loss_boundary'], label='Loss Boundary')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].grid(True, which="both", ls=":")
    axes[1].plot(torch.arange(start=0, end=len(losses['loss_training'])).numpy(),   losses['loss_training'], label='Training loss')
    axes[1].plot(torch.arange(start=0, end=len(losses['loss_validation'])).numpy(), losses['loss_validation'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(folder+'/Losses.png')
    plt.close(fig)


#Save list of training steps for preconditioning
save_list=True
if save_list:
    hfile = h5py.File(folder+"/Training_error.h5", "w")
    hfile.create_dataset("PDE error", data=losses['loss_PDE'])
    hfile.create_dataset("Boundary error", data=losses['loss_boundary'])
    hfile.create_dataset("Train error", data=losses['loss_training'])
    hfile.create_dataset("Validation error", data=losses['loss_validation'])
    hfile.create_dataset("Boundary decay", data=Boundary_decay)
    hfile.close()

print('Finish')


    

