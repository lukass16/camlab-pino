import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from CNOModule import CNO
from torch.utils.data import Dataset

import scipy

from training.FourierFeatures import FourierFeatures, FourierFeaturesPrecondition


# Set the random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Set the data folder path
data_folder = "/cluster/home/lkellijs/camlab-pino/data/"

#------------------------------------------------------------------------------

# Some functions needed for loading the Navier-Stokes data

def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real

def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

#------------------------------------------------------------------------------

#Load default parameters:
    
def default_param(network_properties):
    
    if "channel_multiplier" not in network_properties:
        network_properties["channel_multiplier"] = 32
    
    if "half_width_mult" not in network_properties:
        network_properties["half_width_mult"] = 1
    
    if "lrelu_upsampling" not in network_properties:
        network_properties["lrelu_upsampling"] = 2

    if "filter_size" not in network_properties:
        network_properties["filter_size"] = 6
    
    if "out_size" not in network_properties:
        network_properties["out_size"] = 1
    
    if "radial" not in network_properties:
        network_properties["radial_filter"] = 0
    
    if "cutoff_den" not in network_properties:
        network_properties["cutoff_den"] = 2.0001
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
         network_properties["retrain"] = 4
    
    if "kernel_size" not in network_properties:
        network_properties["kernel_size"] = 3
    
    if "activation" not in network_properties:
        network_properties["activation"] = 'cno_lrelu'
    
    return network_properties

#------------------------------------------------------------------------------

#NOTE:
#All the training sets should be in the folder: data/

#------------------------------------------------------------------------------
#Navier-Stokes data:
#   From 0 to 750 : training samples (750)
#   From 1024 - 128 - 128 to 1024 - 128 : validation samples (128)
#   From 1024 - 128 to 1024 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class ShearLayerDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 750, s=64, in_dist = True):
        
        self.s = s
        self.in_dist = in_dist
        #The file:
        
        if in_dist:
            if self.s==64:
                self.file_data = "data/NavierStokes_64x64_IN.h5" #In-distribution file 64x64               
            else:
                self.file_data = "data/NavierStokes_128x128_IN.h5"   #In-distribution file 128x128
        else:
            self.file_data = "data/NavierStokes_128x128_OUT.h5"  #Out-of_-distribution file 128x128
        
        self.reader = h5py.File(self.file_data, 'r') 
        self.N_max = 1024

        self.n_val  = 128
        self.n_test = 128
        self.min_data = 1.4307903051376343
        self.max_data = -1.4307903051376343
        self.min_model = 2.0603253841400146
        self.max_model= -2.0383243560791016
        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = self.n_val
            self.start = self.N_max - self.n_val - self.n_test
        elif which == "test":
            self.length = self.n_test
            self.start = self.N_max  - self.n_test
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        if self.s == 64 and self.in_dist:
            inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
            labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        else:
            
            inputs = self.reader['Sample_' + str(index + self.start)]["input"][:].reshape(1,1,self.s, self.s)
            labels = self.reader['Sample_' + str(index + self.start)]["output"][:].reshape(1,1, self.s, self.s)
            
            if self.s<128:
                inputs = downsample(inputs, self.s).reshape(1, self.s, self.s)
                labels = downsample(labels, self.s).reshape(1, self.s, self.s)
            else:
                inputs = inputs.reshape(1, 128, 128)
                labels = labels.reshape(1, 128, 128)
            
            inputs = torch.from_numpy(inputs).type(torch.float32)
            labels = torch.from_numpy(labels).type(torch.float32)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid
    
class ShearLayer:
    def __init__(self, network_properties, device, batch_size, training_samples, size, in_dist = True):

        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,     # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        
        num_workers = 0
        self.train_loader = DataLoader(ShearLayerDataset("training", self.N_Fourier_F, training_samples, size), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(ShearLayerDataset("validation", self.N_Fourier_F, training_samples, size), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(ShearLayerDataset("test", self.N_Fourier_F, training_samples, size, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Poisson data:
#   From 0 to 1024 : training samples (1024)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class SinFrequencyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 1024, s=64, in_dist = True):
        
        
        #The file: #! note: these have been updated from the original: PoissonData_PDEDomain2.h5
        if in_dist:
            self.file_data = '/cluster/home/lkellijs/camlab-pino/data/PoissonData_64x64_IN.h5'
        else:
            self.file_data = '/cluster/home/lkellijs/camlab-pino/data/PoissonData_64x64_OUT.h5'

        #Load normalization constants from the TRAINING set:
        self.reader = h5py.File(self.file_data, 'r')
        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]

        self.s = s #Sampling rate

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024+128
            else:
                self.length = 256
                self.start = 0 
        
        #Load different resolutions
        if s!=64:
            self.file_data = "data/Poisson_res/PoissonData_NEW_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed.
        self.reader = h5py.File(self.file_data, 'r')
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        self.N_Fourier_Precondioning=0
        # Try to read Domain from file, otherwise use default value
        try:
            self.Domain = self.reader['Domain']
        except KeyError:
            print("Domain not found in file, using default value of 1.0")
            self.Domain = 1.0  # Default domain size for most Poisson problems
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        # Normalize inputs and labels to [0, 1]
        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        if self.N_Fourier_Precondioning>0:
            #We make fourier transformation of input.
            FF=FourierFeaturesPrecondition(self.N_Fourier_Precondioning, inputs.device)
            inputs=FF(inputs).unsqueeze(0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, self.Domain, self.s)
        y = torch.linspace(0, self.Domain, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid
    
    def get_max_and_min(self):
        Normalization_values= {
        "min_data": self.min_data,
        "max_data": self.max_data,
        "min_model":self.min_model,
        "max_model":self.max_model
         }
        return Normalization_values


class SinFrequency:
    '''This class is used to load the Poisson data
    It returns the input and output data for the Poisson equation.
    
    It contains:
    - self.model: the CNO model
    - self.train_loader: the training loader
    - self.val_loader: the validation loader
    - self.test_loader: the test loader
    '''
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            # !DEBUG
            print(f"in_size: {self.in_size}")
            # !DEBUG END
            print(f"type(in_size): {type(self.in_size)}")
            assert self.in_size<=128 and self.in_size >= 64  # Support 64x64 and 128x128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,     # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------
        

        #Change number of workers accoirding to your preference
        num_workers = 16

        self.train_loader = DataLoader(SinFrequencyDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(SinFrequencyDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(SinFrequencyDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)




#Helmoltz data
class HelmholtzDataset(Dataset):
    def __init__(self, which="training", nf = 0, training_samples = 1024, in_dist = True, N_max =19675, cluster = True,pad_factor=0):
        
        #Dataset 1 (less scales --> up to 8):
        """
        self.file_data = data_folder + "HelmotzData_FixedBC1_4shapes_fixed_w_processed.h5"
        self.reader = h5py.File(self.file_data, 'r')
        
        self.mean = 0.24185410524000955
        self.std  = 1.6480482205609408
        self.N_max = 9404
        """
        
        # Dataset 9k (more scales): 
        #    ---  BC = 0.5, w = 5pi/2,         
        #    ---  n_spots = np.random.randint(2,max_spots+1)
        #    ---  centers = 0.5+ 0.2*2*(np.random.uniform(0,1,(n_spots, 2)))
        #    ---  radius = 0.075 + 0.05*np.random.rand(1, )[0]
        #    ---  which_shape = np.random.randint(1, 3, n_spots)
        #    ---  Scale up to 50
        
        # Dataset 19k: 
        #    ---  BC = uniform(0.25,0.5), w = 5pi/2, spots 2-8 --> Random Gaussian
        #         --- amplitude = 1
        #         --- sigma = np.random.uniform(5, 15)   
        #         --- x_center = np.random.randint(grid_size // 4, 3 * grid_size // 4)
        #         --- y_center = np.random.randint(grid_size // 4, 3 * grid_size // 4)
        #    ---  Scale up to 5
        
        
        self.N_max = N_max
        self.which = which
        self.pad_factor=pad_factor
        if self.N_max == 9931:
            self.mean = 0.236221626070
            self.std = 2.147599637905932
            if cluster:
                self.file_data = data_folder + "HelmotzData_FixedBC1_4shapes_fixed_w_processed_2.h5"
            else:
                self.file_data = data_folder + "HelmotzData_FixedBC1_4shapes_fixed_w_processed_2.h5"

        elif self.N_max == 19675:
            self.mean = 0.11523915668552
            self.std = 0.8279975746000605
            if cluster:
                self.file_data = data_folder + "HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5"
            else:
                self.file_data = data_folder + "HelmotzData_VaryingBC02501_2to8bumbs_w5over2pi_processed.h5"

        self.reader = h5py.File(self.file_data, 'r')        

        assert training_samples < self.N_max - 256 - 256
        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 256
            self.start = self.N_max - 256 - 256
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = self.N_max - 256
            else:
                self.length = 128
                self.start = 0

        #If the reader changed:
        self.reader = h5py.File(self.file_data, 'r') 
        
        #Default:
        self.N_Fourier_F = nf
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["a"][:]).type(torch.float32).reshape(1, 128, 128)
        inputs = inputs-1
        if self.pad_factor!=0:
           #inputs=self.transform(input=inputs,pad_factor=self.pad_factor)
           inputs=self.transform_zero(input=inputs,pad_factor=self.pad_factor)
           
        labels = self.reader['Sample_' + str(index + self.start)]["u"][:]
        labels = (labels - self.mean)/self.std
        labels = torch.from_numpy(labels).type(torch.float32).reshape(1, 128, 128)
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
        
        b     =  torch.from_numpy(np.array(self.reader['Sample_' + str(index + self.start)]["bc"])).to(torch.float32)
        bc    =  torch.ones_like(inputs)*b
        #bc    =  torch.zeros_like(inputs)
        #bc[...,0,:]  = b
        #bc[...,-1,:] = b
        #bc[...,:,0]  = b
        #bc[...,:,-1] = b
        inputs = torch.cat((inputs, bc), 0)
        
        return inputs, labels
    
    def transform(self,input,pad_factor):
        #boundary_pad
        shape=input.shape
        input_pad=torch.ones((*shape[0:-2],shape[-2]+pad_factor,shape[-1]+pad_factor))*input[0,0,0]
        input_pad[...,:shape[-2],:shape[-1]]=input
        return input_pad
    def transform_zero(self,input,pad_factor):
        #boundary_pad
        shape=input.shape
        input_pad=torch.zeros((*shape[0:-2],shape[-2]+pad_factor,shape[-1]+pad_factor))
        input_pad[...,:shape[-2],:shape[-1]]=input
        return input_pad
        
    def get_max_and_min(self):
        Normalization_values= {
        "mean_data": 1,
        "mean_model":self.mean ,
        "std_model": self.std
         }
        return Normalization_values    
    
    def get_grid(self):
        s = 128
        x = torch.linspace(0, 1, s)
        y = torch.linspace(0, 1, s)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class Helmholtz:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 128, in_dist = True, N_max =19788, cluster = True,pad_factor=0):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            #assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        cutoff_den =network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult =network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------

        torch.manual_seed(retrain)
         
        in_dim = 2
        
        self.model = CNO(in_dim  = in_dim + 2*self.N_Fourier_F,     # Number of input channels.
                        in_size = self.in_size,                     # Input spatial size
                        N_layers = N_layers,                        # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                              # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------
        
        #Change number of workers accoirding to your preference
        num_workers = 16
        #   which="training", nf = 0, training_samples = 1024, in_dist = True, N_max =19788, cluster = True):

        self.train_loader = DataLoader(HelmholtzDataset(which = "training", 
                                                        nf = self.N_Fourier_F, 
                                                        training_samples= training_samples, 
                                                        in_dist=in_dist,
                                                        N_max= N_max,
                                                        cluster=cluster,pad_factor=pad_factor), 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers)
        
        self.val_loader = DataLoader(HelmholtzDataset(which = "validation", 
                                                        nf = self.N_Fourier_F, 
                                                        training_samples= training_samples, 
                                                        in_dist=in_dist,
                                                        N_max= N_max,
                                                        cluster=cluster,pad_factor=pad_factor), 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        num_workers=num_workers)

        self.test_loader = DataLoader(HelmholtzDataset(which = "test", 
                                                        nf = self.N_Fourier_F, 
                                                        training_samples= training_samples, 
                                                        in_dist=in_dist,
                                                        N_max= N_max,
                                                        cluster=cluster,pad_factor=pad_factor), 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        num_workers=num_workers)
        

#------------------------------------------------------------------------------
# Wave data
#   From 0 to 512 : training samples (512)
#   From 1024 to 1024 + 128 : validation samples (128)
#   From 1024 + 128 to 1024 + 128 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class WaveEquationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, t = 5, s = 64, in_dist = True):
        
        #Default file:       
        if in_dist:
            self.file_data = "data/WaveData_64x64_IN.h5"
        else:
            self.file_data = "data/WaveData_64x64_OUT.h5"

        self.reader = h5py.File(self.file_data, 'r')
        
        #Load normaliation constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
        
        #What time? DEFAULT : t = 5
        self.t = t
                        
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 1024
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 1024 + 128
            else:
                self.length = 256
                self.start = 0
        
        self.s = s
        if s!=64:
            self.file_data = "data/WaveData_24modes_s" + str(s) + ".h5"
            self.start = 0
        
        #If the reader changed:
        self.reader = h5py.File(self.file_data, 'r') 
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["input"][:]).type(torch.float32).reshape(1, self.s, self.s)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)+"_t_"+str(self.t)]["output"][:]).type(torch.float32).reshape(1, self.s, self.s)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        grid = torch.zeros((self.s, self.s,2))

        for i in range(self.s):
            for j in range(self.s):
                grid[i, j][0] = i/(self.s - 1)
                grid[i, j][1] = j/(self.s - 1)
                
        return grid


class WaveEquation:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024, s = 64, in_dist = True):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #Change number of workers accoirding to your preference
        num_workers = 0
        
        self.train_loader = DataLoader(WaveEquationDataset("training", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(WaveEquationDataset("validation", self.N_Fourier_F, training_samples, 5, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(WaveEquationDataset("test", self.N_Fourier_F, training_samples, 5, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Allen-Cahn data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class AllenCahnDataset(Dataset):
    def __init__(self, which="training", nf = 0, training_samples = 256, s=64, in_dist = True):

        #Default file:
        if in_dist:
            self.file_data = "data/AllenCahn_64x64_IN.h5"
        else:            
            self.file_data = "data/AllenCahn_64x64_OUT.h5"
        self.reader = h5py.File(self.file_data, 'r')
        
        #Load normalization constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]
                
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = 256
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 256 + 128
            else:
                self.length = 128
                self.start = 0 
        
        #Default:
        self.N_Fourier_F = nf
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)
        #print(inputs.shape)            
        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class AllenCahn:
    def __init__(self, network_properties, device, batch_size, training_samples = 1024,  s = 64, in_dist = True):
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------
        

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(AllenCahnDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AllenCahnDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(AllenCahnDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Smooth Transport data
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class ContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized        
        #Default file:       
        if in_dist:
            self.file_data = "data/ContTranslation_64x64_IN.h5"
        else:
            self.file_data = "data/ContTranslation_64x64_OUT.h5"
        
        print(self.file_data)
        self.reader = h5py.File(self.file_data, 'r') 

        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512 + 256
            else:
                self.length = 256
                self.start = 0

        #Default:
        self.N_Fourier_F = nf
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class ContTranslation:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(ContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(ContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(ContTranslationDataset("test", self.N_Fourier_F, training_samples, s,in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Discontinuous Transport data
#   From 0 to 512 : training samples (512)
#   From 512 to 512 + 256 : validation samples (256)
#   From 512 + 256 to 512 + 256 + 256 : test samples (256)
#   Out-of-distribution testing samples: 0 to 256 (256)

class DiscContTranslationDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 64, in_dist = True):
        
        #The data is already normalized
        
        if in_dist:
            self.file_data = "data/DiscTranslation_64x64_IN.h5"
        else:
            self.file_data = "data/DiscTranslation_64x64_OUT.h5"
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 256
            self.start = 512
        elif which == "test":
            if in_dist:
                self.length = 256
                self.start = 512+256
            else:
                self.length = 256
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)


        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class DiscContTranslation:
    def __init__(self, network_properties, device, batch_size, training_samples = 512, s = 64, in_dist = True):
        
        
        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")

        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")

        if "N_res" in network_properties:
            N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")

        if "N_res_neck" in network_properties:
            N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")

        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)

        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]

        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]

        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)


        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(DiscContTranslationDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DiscContTranslationDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(DiscContTranslationDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Compressible Euler data
#   From 0 to 750 : training samples (750)
#   From 750 to 750 + 128 : validation samples (128)
#   From 750 + 128 to 750 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class AirfoilDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples = 512, s = 128, in_dist = True):
        
        #We DO NOT normalize the data in this case
        
        if in_dist:
            self.file_data = "data/Airfoil_128x128_IN.h5"
        else:
            self.file_data = "data/Airfoil_128x128_OUT.h5"
        
        #in_dist = False
        
        if which == "training":
            self.length = training_samples
            self.start = 0
            
        elif which == "validation":
            self.length = 128
            self.start = 750
        elif which == "test":
            if in_dist:
                self.length = 128
                self.start = 750 + 128
            else:
                self.length = 128
                self.start = 0

        self.reader = h5py.File(self.file_data, 'r') 

        #Default:
        self.N_Fourier_F = nf
        
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 128, 128)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 128, 128)
        
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 128)
        y = torch.linspace(0, 1, 128)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class Airfoil:
    def __init__(self, network_properties, device, batch_size, training_samples = 512, s = 128, in_dist = True):
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------
        
        torch.manual_seed(retrain)
        
        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                # Input spatial size
                        N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                         # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)
        
        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(AirfoilDataset("training", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(AirfoilDataset("validation", self.N_Fourier_F, training_samples, s), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(AirfoilDataset("test", self.N_Fourier_F, training_samples, s, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Darcy Flow data
#   From 0 to 256 : training samples (256)
#   From 256 to 256 + 128 : validation samples (128)
#   From 256 + 128 to 256 + 128 + 128 : test samples (128)
#   Out-of-distribution testing samples: 0 to 128 (128)

class DarcyDataset(Dataset):
    def __init__(self, which="training", nf=0, training_samples=256, insample=True):
        
        if insample:
            self.file_data = "data/Darcy_64x64_IN.h5"
        else:
            self.file_data = "data/Darcy_64x64_IN.h5"
        
        
        self.reader = h5py.File(self.file_data, 'r')

        self.min_data = self.reader['min_inp'][()]
        self.max_data = self.reader['max_inp'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
                
        if which == "training":
            self.length = training_samples
            self.start = 0
        elif which == "validation":
            self.length = 128
            self.start = training_samples
        elif which == "testing":
            if insample:
                self.length = 128
                self.start = training_samples + 128
            else:
                self.length = 128
                self.start = 0

        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data) / (self.max_data - self.min_data)
        labels = (labels - self.min_model) / (self.max_model - self.min_model)

        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            grid = grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, grid), 0)

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, 64)
        y = torch.linspace(0, 1, 64)

        x_grid, y_grid = torch.meshgrid(x, y)

        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)

        if self.N_Fourier_F > 0:
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            grid = FF(grid)
        return grid
    
class Darcy:
    def __init__(self, network_properties, device, batch_size, training_samples = 512,  s = 64, in_dist = True):
        
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
            assert self.in_size<=128        
        else:
            raise ValueError("You must specify the computational grid size.")

        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
            N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")

        if "N_res_neck" in network_properties:
            N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")

        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)

        #----------------------------------------------------------------------
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
        activation = network_properties["activation"]
        ##----------------------------------------------------------------------

        torch.manual_seed(retrain)

        self.model = CNO(in_dim  = 1 + 2*self.N_Fourier_F,      # Number of input channels.
                        in_size = self.in_size,                 # Input spatial size
                        N_layers = N_layers,                    # Number of (D) and (U) Blocks in the network
                        N_res = N_res,                          # Number of (R) Blocks per level
                        N_res_neck = N_res_neck,
                        channel_multiplier = channel_multiplier,
                        conv_kernel=kernel_size,
                        cutoff_den = cutoff_den,
                        filter_size=filter_size,  
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult  = half_width_mult,
                        activation = activation).to(device)

        #----------------------------------------------------------------------

        #Change number of workers accoirding to your preference
        num_workers = 0

        self.train_loader = DataLoader(DarcyDataset("training", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(DarcyDataset("validation", self.N_Fourier_F, training_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(DarcyDataset("testing", self.N_Fourier_F, training_samples, in_dist), batch_size=batch_size, shuffle=False, num_workers=num_workers)
