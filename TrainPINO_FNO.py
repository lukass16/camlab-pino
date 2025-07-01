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

from Problems.FNOBenchmarks import SinFrequency, Helmholtz
from Physics_NO.loss_functions.Relative_loss import Relative_loss
from Physics_NO.loss_functions.ModulePDELoss import Loss_PDE,Loss_OP



"""-------------------------------Setting parameters for training--------------------------------"""
'''
Currently the code trains an FNO with optional pretraining
'''
MODEL_DIR = "helmholtz_debug"
if len(sys.argv) == 2:

    InfoPretrainedNetwork= {
        #----------------------------------------------------------------------
        #Load Trained model: (Must be compatible with model_architecture)
        #Path to pretrained model: None for training from scratch
        "Path to pretrained model": f"TrainedModels/{MODEL_DIR}/FNO_1024helmholtz",
        "Pretrained Samples":  1024,
    }

    training_properties = {
       "learning_rate": 3e-5, 
        "weight_decay": 1e-10,
        "scheduler_step": 10, # number of steps after which the learning rate is decayed
        "scheduler_gamma": 0.98,
        "epochs": 30,   #! changed for the test
        "batch_size": 16,
        "exp": 3,                # Do we use L1 or L2 errors? Default: L1 3 for smooth
        "training_samples": 1024,  # How many training samples?
        "lambda": 100,
        "boundary_weight":1,
        "pad_factor": 0,
        "patience": 1.0, #patience for early stopping  - usually 0.4 #! changed for the test
        "gradient_clip_value": 5 # set None for no gradient clipping
    }

    # FNO architecture (only used when training from scratch)
    fno_architecture_ = {
        "width": 128, # Resolution of the computational grid
        "modes": 16,
        "FourierF" : 0, #Number of Fourier Features in the input channels. Default is 0.
        "n_layers": 4, #Number of Fourier layers
        "padding": 0,
        "include_grid":1,
        "retrain": 4, #Random seed
    }
   
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   helmholtz           : Helmholtz equation
    
    which_example = sys.argv[1]
    
    CUSTOM_FLAG = "_start_debug_grad_norm_10_lambda_100_boundary_weight_1" #! changed for the test

    # if pretrained
    if InfoPretrainedNetwork["Path to pretrained model"] is not None:
        folder = "TrainedModels/"+which_example+"/PINO+_FNO_pretrained"+which_example+CUSTOM_FLAG
    else:
        folder = "TrainedModels/"+which_example+"/PINO+_FNO_no_pretraining"+which_example+CUSTOM_FLAG
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    fno_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    InfoPretrainedNetwork = json.loads(sys.argv[4].replace("\'", "\""))
    if InfoPretrainedNetwork["Path to pretrained model"]=='None':
       InfoPretrainedNetwork["Path to pretrained model"]=None
    which_example = sys.argv[5]


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
lampda=training_properties['lambda']
boundary_weight=training_properties['boundary_weight']
pad_factor=training_properties['pad_factor']
gradient_clip_value = training_properties.get("gradient_clip_value")


if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)


"""------------------------------------Load  pretrained model and data--------------------------------------"""
# save the training properties and the architecture of the pretrained model
df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')

if InfoPretrainedNetwork["Path to pretrained model"] is not None:
    df = pd.DataFrame.from_dict([InfoPretrainedNetwork]).T
    df.to_csv(folder + '/InfoPretrainedNetwork.txt', header=False, index=True, mode='w')
    net_architucture_path = InfoPretrainedNetwork["Path to pretrained model"]+'/net_architecture.txt'
    df = pd.read_csv(net_architucture_path, header=None, index_col=0)
    fno_architecture_ = df.to_dict()[1]
    fno_architecture_ = {key: int(value) if str(value).isdigit() else float(value) if '.' in str(value) else value for key, value in df.to_dict()[1].items()}
    df = pd.DataFrame.from_dict([fno_architecture_]).T
    df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

in_size=fno_architecture_["width"]

# load the data
if which_example == "poisson":
    example = SinFrequency(fno_architecture_, device, batch_size,training_samples)
elif which_example == "helmholtz":
    example = Helmholtz(fno_architecture_, device, batch_size, training_samples)
else:
    raise ValueError("Dataset type not found. Please choose between {poisson, helmholtz}")

# create the model folder
if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

# save the training properties and the architecture
df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([fno_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

# load the model
model = example.model

if InfoPretrainedNetwork["Path to pretrained model"] is not None:
    pretrained_model_path = InfoPretrainedNetwork["Path to pretrained model"]+'/model.pkl'
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"The model file '{pretrained_model_path}' does not exist.")

    # 1 load the model for finetuning
    model = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    # fix device mismatch
    if hasattr(model, 'device'):
        model.device = device
    model = model.to(device)

    # 2 load the anchor model (for the anchor loss)
    model_fix=torch.load(pretrained_model_path, map_location=device, weights_only=False)
    # fix device mismatch
    if hasattr(model_fix, 'device'):
        model_fix.device = device
    model_fix = model_fix.to(device)

    for param in model_fix.parameters():
       param.requires_grad = False

    print(f'Loading trained network from {pretrained_model_path}')
else:
    print(f'Network is trained from scratch')
    df = pd.DataFrame.from_dict([fno_architecture_]).T
    df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')


"""------------------------------------Train--------------------------------------"""
n_params = model.print_size()
train_loader = example.train_loader
val_loader = example.val_loader

Normalization_values=train_loader.dataset.get_max_and_min() # need these for PI-loss

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

#! DEBUG - test different lr schedulers
# steps_per_epoch = len(train_loader)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=learning_rate,           # peak LR you want to hit
#     epochs=epochs,
#     steps_per_epoch=steps_per_epoch,
#     pct_start=0.1,         # fraction of cycle spent increasing LR
#     anneal_strategy='cos', # cosine decay after the peak
#     final_div_factor=1e4,  # LR at the end = max_lr / final_div_factor
# )


if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

# define the loss functions
loss_relative=Relative_loss(pad_factor,in_size)
loss_pde     =Loss_PDE(which_example=which_example,Normalization_values=Normalization_values,p=p,\
                       pad_factor=pad_factor,in_size=in_size)
Operator_loss=Loss_OP(p=p,in_size=in_size,pad_factor=pad_factor) # this is the anchor loss
losses = {'loss_PDE': [],'loss_boundary': [], 'loss_OP': [], 'loss_training': [], 'loss_validation':[]}


patience = int(training_properties["patience"] * epochs)
best_model_testing_error = 300
counter = 0

for epoch in range(epochs):
    
    # each epoch
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse, train_op, train_f = 0.0, 0.0, 0.0
        losses['loss_PDE'].append(0)
        losses['loss_OP'].append(0)
        losses['loss_boundary'].append(0)
        running_relative_train_mse = 0.0
        
        
        #! DEBUG - start with validation
        """------------------------------------Validation--------------------------------------"""
        # after each epoch, we evaluate the model on the validation and train set       
        # we only calculate relative error for these
        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0

            # loop through the validation loader
            for step, (input_batch, output_batch) in enumerate(val_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                loss_f=loss_relative(output_pred_batch.view(batch_size,-1,in_size,in_size),output_batch.view(batch_size,-1,in_size,in_size))

                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)
            losses['loss_validation'].append(test_relative_l2)

            # loop through the training loader
            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    if which_example == "airfoil":
                        output_pred_batch[input_batch==1] = 1
                        output_batch[input_batch==1] = 1
                    
                    loss_f = loss_relative(output_pred_batch.view(batch_size,-1,in_size,in_size),output_batch.view(batch_size,-1,in_size,in_size))
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader) # take the average of the losses
            losses['loss_training'].append(train_relative_l2)
            
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
                
            # write debug file #! DEBUG
            epoch_num = epoch
            with open(folder + '/debug.txt', 'a') as file:
                file.write(f"Epoch: {epoch_num}\n")
                file.write(f"Train loss: {train_relative_l2}\n")
                file.write(f"Validation loss: {test_relative_l2}\n")
                file.write(f"Best model testing error: {best_model_testing_error}\n")
                
        """------------------------------------------------------------------------------------------------"""

        
        # loop through the training loader
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)
            
            # get the loss (convert to shapes (B, 2, H, W) and (B, 1, H, W))
            loss_PDE,loss_boundary= loss_pde(input=input_batch.view(batch_size,-1,in_size,in_size),\
                                             output=output_pred_batch.view(batch_size,-1,in_size,in_size))
            loss_f=loss_PDE+boundary_weight*loss_boundary
            
            if InfoPretrainedNetwork["Path to pretrained model"] is not None:
                # get the anchor output
                with torch.no_grad():
                     output_fix=model_fix(input_batch)
                
                loss_op=Operator_loss(output_train=output_pred_batch.view(batch_size,-1,in_size,in_size),\
                                      output_fix=output_fix.view(batch_size,-1,in_size,in_size))
                loss_total=loss_op*lampda+loss_f
                losses['loss_OP'][-1] += loss_op.item()*lampda
            else:
                loss_total=loss_f
                losses['loss_OP'][-1] += 0.0

            losses['loss_PDE'][-1]     +=loss_PDE.item()       #values for plot
            losses['loss_boundary'][-1]+=boundary_weight*loss_boundary.item()  #values for plot
            loss_total.backward()
            
            # optional gradient clipping
            if gradient_clip_value is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            else:
                total_norm = 0.0
            
            #! DEBUG - log the gradient norm and the learning rate
            current_lr = optimizer.param_groups[0]['lr']
            log_mode = 'a'
            # On the first step of the first epoch, overwrite the log file.
            if epoch == 0 and step == 0:
                log_mode = 'w'
            with open(os.path.join(folder, 'optim_debug.txt'), log_mode) as f:
                f.write(f"Epoch {epoch}, Step {step}: Total Gradient Norm = {total_norm}, Learning Rate = {current_lr}\n")
            
            optimizer.step()
            scheduler.step() # Step the scheduler after each batch for OneCycleLR
            train_mse = train_mse * step / (step + 1) + loss_total.item() / (step + 1)
            train_op = train_op * step / (step + 1) + losses['loss_OP'][-1]/ (step + 1)
            train_f = train_f * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse,'Operator loss': train_op,'PDE and Boundary loss': train_f})

        losses['loss_OP'][-1]/=len(train_loader)
        losses['loss_PDE'][-1]/=len(train_loader)
        losses['loss_boundary'][-1]/=len(train_loader)     
        
        # save the losses
        writer.add_scalar("train_loss/train_loss", train_mse, epoch)
        # save the individual losses
        writer.add_scalar("train_loss/train_loss_pde", losses['loss_PDE'][-1], epoch)
        writer.add_scalar("train_loss/train_loss_boundary", losses['loss_boundary'][-1], epoch)
        writer.add_scalar("train_loss/train_loss_op", losses['loss_OP'][-1], epoch)

        

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()
        
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
            
    
    # early stopping
    if counter>patience:
        print("Early Stopping")
        break

    # plot the losses
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].grid(True, which="both", ls=":")
    axes[0].plot(torch.arange(start=0, end=len(losses['loss_PDE'])).numpy(), losses['loss_PDE'], label='Loss PDE')
    axes[0].plot(torch.arange(start=0, end=len(losses['loss_boundary'])).numpy(), losses['loss_boundary'], label='Loss Boundary')
    axes[0].plot(torch.arange(start=0, end=len(losses['loss_OP'])).numpy(), losses['loss_OP'], label='Loss OP')
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

print('Finish')


    

