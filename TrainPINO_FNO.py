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
from Physics_NO.loss_functions.ModulePDELoss import Loss_PDE,Loss_OP, Laplace, Unnormalize
from FiniteDifferences import Laplace as FDLaplace

def create_helmholtz_evolution_plot(input_batch, output_batch, output_pred_batch, in_size, 
                                   step_number, folder, Normalization_values, device):
    """
    Create evolution plots for Helmholtz equation showing true/predicted labels, laplacians, and targets.
    Uses finite difference laplacian computation.
    
    Args:
        input_batch: Input batch (B, 2, H, W) where channels are (a, boundary) for Helmholtz
        output_batch: True labels (B, 1, H, W)  
        output_pred_batch: Predicted labels (B, 1, H, W)
        in_size: Grid size
        step_number: Current optimization step
        folder: Directory to save plots
        Normalization_values: For unnormalization
        device: torch device
    """
    
    # Create evolution_plots directory if it doesn't exist
    plots_dir = os.path.join(folder, "evolution_plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Take the first sample from the batch
    input_sample = input_batch[0:1]
    label_sample = output_batch[0:1] 
    pred_sample = output_pred_batch[0:1]
    
    # Unnormalize the data
    unnormalize_fn = Unnormalize("helmholtz", Normalization_values)
    input_unnorm, label_unnorm = unnormalize_fn(input=input_sample, output=label_sample)
    _, pred_unnorm = unnormalize_fn(input=input_sample, output=pred_sample)
    
    # Convert to (B, H, W) format for laplacian computation
    label_unnorm_2d = label_unnorm.squeeze(1)  # (1, H, W)
    pred_unnorm_2d = pred_unnorm.squeeze(1)    # (1, H, W)
    
    # Compute laplacians using finite differences
    laplace_fn = FDLaplace(s=in_size, D=1.0)
    lap_label, cut_size = laplace_fn(label_unnorm_2d)
    lap_pred, _ = laplace_fn(pred_unnorm_2d)
    
    # For Helmholtz: target = -ω²a²u
    omega = 5 * torch.pi / 2  # Default omega value from loss function
    a = input_unnorm[0, 0, :, :]  # Coefficient field
    
    # Crop a to match laplacian size
    a_cropped = a[cut_size:-cut_size, cut_size:-cut_size]
    
    # Compute targets
    target_label = -omega**2 * a_cropped**2 * label_unnorm_2d[0, cut_size:-cut_size, cut_size:-cut_size]
    target_pred = -omega**2 * a_cropped**2 * pred_unnorm_2d[0, cut_size:-cut_size, cut_size:-cut_size]
    
    # Convert to numpy for plotting
    def to_numpy(x):
        return x.detach().cpu().numpy()
    
    # Get the data for plotting (crop to match laplacian size)
    label_plot = to_numpy(label_unnorm[0, 0, cut_size:-cut_size, cut_size:-cut_size])
    pred_plot = to_numpy(pred_unnorm[0, 0, cut_size:-cut_size, cut_size:-cut_size])
    lap_label_plot = to_numpy(lap_label[0])
    lap_pred_plot = to_numpy(lap_pred[0])
    target_label_plot = to_numpy(target_label)
    target_pred_plot = to_numpy(target_pred)
    
    # Create the plot with space for horizontal colorbars
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Predictions before step {step_number}", fontsize=16)
    
    # Create a grid: 3 rows (2 for plots, 1 for colorbars) x 3 columns
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.1], hspace=0.3, wspace=0.3)
    
    # Create subplot axes for the plots (2x3 grid)
    axes = []
    for i in range(2):
        row = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axes.append(row)
    
    # Calculate shared color scales as requested
    # For u and u* (column 0)
    vmin_u = min(label_plot.min(), pred_plot.min())
    vmax_u = max(label_plot.max(), pred_plot.max())
    
    # For laplacians (column 1)
    vmin_lap = min(lap_label_plot.min(), lap_pred_plot.min())
    vmax_lap = max(lap_label_plot.max(), lap_pred_plot.max())
    
    # For targets (column 2)
    vmin_target = min(target_label_plot.min(), target_pred_plot.min())
    vmax_target = max(target_label_plot.max(), target_pred_plot.max())
    
    # Row 0: True values
    im00 = axes[0][0].imshow(label_plot, cmap='gist_ncar', vmin=vmin_u, vmax=vmax_u)
    axes[0][0].set_title('True label u')
    
    im01 = axes[0][1].imshow(lap_label_plot, cmap='gist_ncar', vmin=vmin_lap, vmax=vmax_lap)
    axes[0][1].set_title('Laplacian of true label ∇²u')
    
    im02 = axes[0][2].imshow(target_label_plot, cmap='gist_ncar', vmin=vmin_lap, vmax=vmax_lap)
    axes[0][2].set_title('Target using label -ω²a²u')
    
    # Row 1: Predicted values  
    im10 = axes[1][0].imshow(pred_plot, cmap='gist_ncar', vmin=vmin_u, vmax=vmax_u)
    axes[1][0].set_title('Predicted label u*')
    
    im11 = axes[1][1].imshow(lap_pred_plot, cmap='gist_ncar', vmin=vmin_lap, vmax=vmax_lap)
    axes[1][1].set_title('Laplacian of prediction ∇²u*')
    
    im12 = axes[1][2].imshow(target_pred_plot, cmap='gist_ncar', vmin=vmin_lap, vmax=vmax_lap)
    axes[1][2].set_title('Target using prediction -ω²a²u*')
    
    # Add horizontal colorbars below each column
    # Column 0 colorbar (u and u*)
    cax0 = fig.add_subplot(gs[2, 0])
    fig.colorbar(im00, cax=cax0, orientation='horizontal')
    
    # Column 1 colorbar (laplacians)
    cax1 = fig.add_subplot(gs[2, 1])
    fig.colorbar(im01, cax=cax1, orientation='horizontal')
    
    # Column 2 colorbar (targets)
    cax2 = fig.add_subplot(gs[2, 2])
    fig.colorbar(im02, cax=cax2, orientation='horizontal')
    
    # Save the plot
    plot_filename = f"evolution_step_{step_number:06d}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ====================== NEW: Create separate laplacian comparison plot ======================
    
    # Create different Laplacian operators
    laplace_9pt = FDLaplace(s=in_size, D=1.0, type="9-point")  # Standard (default)
    laplace_13pt = FDLaplace(s=in_size, D=1.0, type="13-point")  # 13-point stencil
    laplace_fourier = Laplace(s=in_size, D=1.0)  # Fourier method
    
    # Apply different Laplacians to true label and prediction
    lap_label_9pt, cut_9pt = laplace_9pt(label_unnorm_2d)
    lap_pred_9pt, _ = laplace_9pt(pred_unnorm_2d)
    
    lap_label_13pt, cut_13pt = laplace_13pt(label_unnorm_2d) 
    lap_pred_13pt, _ = laplace_13pt(pred_unnorm_2d)
    
    lap_label_fourier = laplace_fourier(label_unnorm_2d)
    lap_pred_fourier = laplace_fourier(pred_unnorm_2d)
    
    # Crop all results to cut_size=2 (to match 13-point stencil dimensions)
    crop_size = 2
    
    # Crop true label and prediction to match
    label_cropped = label_unnorm_2d[0, crop_size:-crop_size, crop_size:-crop_size]
    pred_cropped = pred_unnorm_2d[0, crop_size:-crop_size, crop_size:-crop_size]
    
    # Crop laplacians to match 13-point size
    lap_label_9pt_cropped = lap_label_9pt[0, (crop_size-cut_9pt):-(crop_size-cut_9pt), (crop_size-cut_9pt):-(crop_size-cut_9pt)]
    lap_pred_9pt_cropped = lap_pred_9pt[0, (crop_size-cut_9pt):-(crop_size-cut_9pt), (crop_size-cut_9pt):-(crop_size-cut_9pt)]
    
    lap_label_13pt_cropped = lap_label_13pt[0]  # Already cropped to cut_size=2
    lap_pred_13pt_cropped = lap_pred_13pt[0]
    
    lap_label_fourier_cropped = lap_label_fourier[0, crop_size:-crop_size, crop_size:-crop_size]
    lap_pred_fourier_cropped = lap_pred_fourier[0, crop_size:-crop_size, crop_size:-crop_size]
    
    # Convert to numpy for plotting
    lap_label_9pt_plot = to_numpy(lap_label_9pt_cropped)
    lap_pred_9pt_plot = to_numpy(lap_pred_9pt_cropped)
    lap_label_13pt_plot = to_numpy(lap_label_13pt_cropped)
    lap_pred_13pt_plot = to_numpy(lap_pred_13pt_cropped)
    lap_label_fourier_plot = to_numpy(lap_label_fourier_cropped)
    lap_pred_fourier_plot = to_numpy(lap_pred_fourier_cropped)
    
    # Use the color scale from the standard 9-point finite difference (plot 2)
    vmin_lap_comp = lap_pred_9pt_plot.min()
    vmax_lap_comp = lap_pred_9pt_plot.max()
    
    # Calculate residuals (absolute differences from true label)
    residual_9pt = to_numpy(torch.abs(lap_label_9pt_cropped - lap_pred_9pt_cropped))
    residual_13pt = to_numpy(torch.abs(lap_label_13pt_cropped - lap_pred_13pt_cropped))  
    residual_fourier = to_numpy(torch.abs(lap_label_fourier_cropped - lap_pred_fourier_cropped))
    residual_true = to_numpy(torch.zeros_like(lap_label_9pt_cropped))  # True vs true = 0
    
    # Calculate relative errors (L2 norm)
    def calc_relative_error(pred, true):
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        return torch.norm(torch.tensor(pred_flat) - torch.tensor(true_flat)).item() / torch.norm(torch.tensor(true_flat)).item()
    
    rel_error_true = 0.0  # True vs true
    rel_error_9pt = calc_relative_error(lap_pred_9pt_plot, lap_label_9pt_plot)
    rel_error_13pt = calc_relative_error(lap_pred_13pt_plot, lap_label_13pt_plot)
    rel_error_fourier = calc_relative_error(lap_pred_fourier_plot, lap_label_fourier_plot)
    
    # Create the laplacian comparison plot with residuals
    fig_lap = plt.figure(figsize=(20, 10))
    fig_lap.suptitle(f"Laplacian Comparison - Step {step_number}", fontsize=16, y=0.95)
    
    # Create subplots: 2 rows (laplacians + residuals), 4 columns, with space for horizontal colorbars
    gs_lap = fig_lap.add_gridspec(4, 4, height_ratios=[1, 1, 0.05, 0.05], hspace=0.4, wspace=0.3)
    
    # Create subplot axes for laplacians (top row)
    axes_lap = []
    for j in range(4):
        ax = fig_lap.add_subplot(gs_lap[0, j])
        axes_lap.append(ax)
    
    # Create subplot axes for residuals (second row)
    axes_res = []
    for j in range(4):
        ax = fig_lap.add_subplot(gs_lap[1, j])
        axes_res.append(ax)
    
    # Plot the laplacians (top row)
    im_lap0 = axes_lap[0].imshow(lap_label_9pt_plot, cmap='gist_ncar', vmin=vmin_lap_comp, vmax=vmax_lap_comp)
    axes_lap[0].set_title('Laplacian of true label\n(9-point FD)', fontsize=12)
    
    im_lap1 = axes_lap[1].imshow(lap_pred_9pt_plot, cmap='gist_ncar', vmin=vmin_lap_comp, vmax=vmax_lap_comp)
    axes_lap[1].set_title('Laplacian of prediction\n(9-point FD standard)', fontsize=12)
    
    im_lap2 = axes_lap[2].imshow(lap_pred_13pt_plot, cmap='gist_ncar', vmin=vmin_lap_comp, vmax=vmax_lap_comp)
    axes_lap[2].set_title('Laplacian of prediction\n(13-point FD)', fontsize=12)
    
    im_lap3 = axes_lap[3].imshow(lap_pred_fourier_plot, cmap='gist_ncar', vmin=vmin_lap_comp, vmax=vmax_lap_comp)
    axes_lap[3].set_title('Laplacian of prediction\n(Fourier method)', fontsize=12)
    
    # Determine color scale for residuals
    all_residuals = [residual_true, residual_9pt, residual_13pt, residual_fourier]
    vmin_res = 0.0  # Absolute values start at 0
    vmax_res = max([res.max() for res in all_residuals])
    
    # Plot the residuals (second row)
    im_res0 = axes_res[0].imshow(residual_true, cmap='Reds', vmin=vmin_res, vmax=vmax_res)
    axes_res[0].set_title('|True - True|', fontsize=12)
    axes_res[0].text(0.05, 0.95, f'Rel. Error: {rel_error_true:.3f}', transform=axes_res[0].transAxes, 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
                     verticalalignment='top')
    
    im_res1 = axes_res[1].imshow(residual_9pt, cmap='Reds', vmin=vmin_res, vmax=vmax_res)
    axes_res[1].set_title('|True - 9pt FD|', fontsize=12)
    axes_res[1].text(0.05, 0.95, f'Rel. Error: {rel_error_9pt:.3f}', transform=axes_res[1].transAxes, 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
                     verticalalignment='top')
    
    im_res2 = axes_res[2].imshow(residual_13pt, cmap='Reds', vmin=vmin_res, vmax=vmax_res)
    axes_res[2].set_title('|True - 13pt FD|', fontsize=12)
    axes_res[2].text(0.05, 0.95, f'Rel. Error: {rel_error_13pt:.3f}', transform=axes_res[2].transAxes, 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
                     verticalalignment='top')
    
    im_res3 = axes_res[3].imshow(residual_fourier, cmap='Reds', vmin=vmin_res, vmax=vmax_res)
    axes_res[3].set_title('|True - Fourier|', fontsize=12)
    axes_res[3].text(0.05, 0.95, f'Rel. Error: {rel_error_fourier:.3f}', transform=axes_res[3].transAxes, 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
                     verticalalignment='top')
    
    # Add horizontal colorbars
    # Colorbar for laplacians
    cax_lap = fig_lap.add_subplot(gs_lap[2, :])
    cbar_lap = fig_lap.colorbar(im_lap1, cax=cax_lap, orientation='horizontal')
    cbar_lap.set_label('Laplacian Values', fontsize=10)
    
    # Colorbar for residuals  
    cax_res = fig_lap.add_subplot(gs_lap[3, :])
    cbar_res = fig_lap.colorbar(im_res1, cax=cax_res, orientation='horizontal')
    cbar_res.set_label('Absolute Residual', fontsize=10)
    
    # Save the laplacian comparison plot
    lap_plot_filename = f"laplacian_{step_number:06d}.png"
    lap_plot_path = os.path.join(plots_dir, lap_plot_filename)
    plt.savefig(lap_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_lap)
    
    # ========================================================================================


"""-------------------------------Setting parameters for training--------------------------------"""
'''
Currently the code trains an FNO with optional pretraining
'''

if len(sys.argv) == 2:

    InfoPretrainedNetwork= {
        #----------------------------------------------------------------------
        #Load Trained model: (Must be compatible with model_architecture)
        #Path to pretrained model: None for training from scratch
        "Path to pretrained model": "TrainedModels/helmholtz/FNO_1024helmholtz",
        "Pretrained Samples":  1024,
    }

    training_properties = {
        "learning_rate": 3e-4, 
        "weight_decay": 1e-10,
        "scheduler_step": 10, # number of steps after which the learning rate is decayed
        "scheduler_gamma": 0.98,
        "epochs": 1,
        "batch_size": 16,
        "exp": 3,                # Do we use L1 or L2 errors? Default: L1 3 for smooth
        "training_samples": 1024,  # How many training samples?
        "lambda": 100,
        "boundary_weight":1, # best known values for helmholtz: 1 for pure, 10 for pretrained (?)
        "pad_factor": 0,
        "patience": 1.0, #patience for early stopping  - usually 0.4 
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
    
    MODEL_DIR = "helmholtz"
    CUSTOM_FLAG = "_plot_laplacian" #! changed for the test

    # if pretrained
    if InfoPretrainedNetwork["Path to pretrained model"] is not None:
        folder = f"TrainedModels/{MODEL_DIR}/PINO+_FNO_pretrained{which_example}{CUSTOM_FLAG}"
    else:
        folder = f"TrainedModels/{MODEL_DIR}/PINO+_FNO_no_pretraining{which_example}{CUSTOM_FLAG}"
        
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
total_steps = 0  # Track total optimization steps across all epochs

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
                
                # reshape
                input_batch_ = input_batch.permute(0,3,1,2)
                output_batch_ = output_batch.permute(0,3,1,2)
                output_pred_batch_ = output_pred_batch.permute(0,3,1,2)
                
                loss_f=loss_relative(output_pred_batch_,output_batch_)

                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)
            losses['loss_validation'].append(test_relative_l2)

            # loop through the training loader
            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    # reshape
                    input_batch_ = input_batch.permute(0,3,1,2)
                    output_batch_ = output_batch.permute(0,3,1,2)
                    output_pred_batch_ = output_pred_batch.permute(0,3,1,2)
                    
                    if which_example == "airfoil":
                        output_pred_batch[input_batch==1] = 1
                        output_batch[input_batch==1] = 1
                    
                    loss_f = loss_relative(output_pred_batch_,output_batch_)
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
            input_batch_ = input_batch.permute(0,3,1,2)  
            output_batch_ = output_batch.permute(0,3,1,2)
            output_pred_batch_ = output_pred_batch.permute(0,3,1,2)
            #! DEBUG
            assert input_batch_.shape == (batch_size, 2, in_size, in_size)
            assert output_batch_.shape == (batch_size, 1, in_size, in_size)
            assert output_pred_batch_.shape == (batch_size, 1, in_size, in_size)

            loss_PDE,loss_boundary= loss_pde(input=input_batch_, output=output_pred_batch_) #! test new loss function label=output_batch_
            loss_f=loss_PDE+boundary_weight*loss_boundary
            
            if InfoPretrainedNetwork["Path to pretrained model"] is not None:
                # get the anchor output
                with torch.no_grad():
                     output_fix=model_fix(input_batch)
                     output_fix_ = output_fix.permute(0,3,1,2) # reshape
                
                loss_op=Operator_loss(output_train=output_pred_batch_,output_fix=output_fix_)
                
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
                
            #! DEBUG start - make the evolution plot (only for Helmholtz)
            if which_example == "helmholtz" and epoch == 0:
                create_helmholtz_evolution_plot(input_batch_, output_batch_, output_pred_batch_, in_size, total_steps, folder, Normalization_values, device)
            #! DEBUG end - make the evolution plot
            
            optimizer.step()
            total_steps += 1  # Increment total steps counter
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


    

