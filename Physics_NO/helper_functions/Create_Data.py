import h5py
import numpy as np
import os


#   "which_example" can be 

#   poisson             : Poisson equation 
#   helmolz             : Helmolz equation

which_example='Poisson'

dist='_out'
#Path to the folder
path_to_Data= '/cluster/scratch/harno/data' 
Name_of_data=f'{which_example}Data_PDE' 
training_samples=1024+128+256
s=64  #grid points
K=16  #Frequncy modes
D=2   #length of domain

if dist=='_out':
    training_samples=256
    K=20
    Name_of_data=Name_of_data+dist+str(K)
    


if not os.path.isdir(path_to_Data):
    print("Generated new folder")
    os.mkdir(path_to_Data)



file_data=os.path.join(path_to_Data,Name_of_data+'Domain'+str(D)+'.h5')

def calculate_f(x, y, K, a, r):
    result = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            result += a[i][j] *  ((i)**2 + (j)**2)**r * np.sin(np.pi * (i) * x) * np.sin(np.pi * (j) * y)
    return (np.pi  / ( K**2))* result


def calculate_u(x, y, K, a, r):
    result = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if (i**2 + j**2) == 0:
                result += 0  
            else:
                result += a[i][j] * ((i)**2 + (j)**2)**(-r) * np.sin(np.pi * (i) * x) * np.sin(np.pi * (j) * y)
    return (1 / (np.pi * K**2)) * result


def evaluate_function(s,K):
   f=np.zeros((s,s))
   u=np.zeros((s,s))
   a = 2 * np.random.rand(K,K) - 1
   r=0.5
 
   for x in range(0,s):
       for y in range(0,s):
           f[x,y]=calculate_f(D*x/s, D*y/s, K, a, r)
           u[x,y]=calculate_u(D*x/s, D*y/s, K, a, r)
   return f,u

print(f'{file_data}')

print('Begin File')
with h5py.File(file_data, 'w') as hf:
    min_inp_val=np.inf
    max_inp_val=-np.inf
    min_out_val=np.inf
    max_out_val=-np.inf
    for index in range(training_samples):
        inputs,outputs = evaluate_function(s,K)
        min_inp_val=min(np.min(inputs),min_inp_val)
        max_inp_val=max(np.max(inputs),max_inp_val)
        min_out_val=min(np.min(outputs),min_out_val)
        max_out_val=max(np.max(outputs),max_out_val)
        if index%50==0:
           print(index)
        # Add the data to the HDF5 file
        hf.create_dataset(f'Sample_{index}/input', data=inputs, dtype=np.float32)
        hf.create_dataset(f'Sample_{index}/output', data=outputs, dtype=np.float32)
    hf.create_dataset('max_inp', data=max_inp_val, dtype=np.float32)
    hf.create_dataset('min_inp', data=min_inp_val, dtype=np.float32)
    hf.create_dataset('max_out', data=max_out_val, dtype=np.float32)
    hf.create_dataset('min_out', data=min_out_val, dtype=np.float32)
    hf.create_dataset('K', data=K, dtype=np.float32)
    hf.create_dataset('Domain', data=D, dtype=np.float32)

if dist=='_out':
    Name_of_data_in=f'{which_example}Data_PDE' 
    file_data_in=os.path.join(path_to_Data,Name_of_data_in+'Domain'+str(D)+'.h5')
    if not os.path.exists(file_data_in): 
        raise FileNotFoundError(f"The file '{file_data_in}' must exist! Create first in distrubtion/trainings data")
    reader = h5py.File(file_data_in, 'r')
    min_data  = reader['min_inp'][()]
    max_data  = reader['max_inp'][()]
    min_model = reader['min_out'][()]
    max_model = reader['max_out'][()]
    
    
    file_data_out = file_data
    with h5py.File(file_data_out, 'a') as hf:
        hf['min_inp'][...] = min_data
        hf['max_inp'][...] = max_data
        hf['min_out'][...] = min_model
        hf['max_out'][...] = max_model

print('Finish File')