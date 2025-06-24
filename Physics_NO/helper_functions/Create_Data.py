import h5py
import numpy as np
import os


#   "which_example" can be 

#   poisson             : Poisson equation 
#   helmolz             : Helmolz equation

which_example='Helmholtz'

dist=''
#Path to the folder
path_to_Data= '/cluster/home/lkellijs/camlab-pino/data'
Name_of_data=f'{which_example}Data_PDE' 
training_samples=1024+128+256
s=64  #grid points
K=16  #Frequncy modes
D=1   #length of domain

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


def calculate_f_helmholtz(x, y, K, a, r, omega, u_val):
    laplacian_u_val = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if (i**2 + j**2) != 0:
                laplacian_u_val -= a[i][j] * ((i)**2 + (j)**2)**(1-r) * np.sin(np.pi * (i) * x) * np.sin(np.pi * (j) * y)
    
    laplacian_u_val *= (np.pi**2 / (np.pi * K**2)) # Scaling factor adjustment
    
    return laplacian_u_val + (omega**2) * u_val

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

def evaluate_function_helmholtz(s, K, omega):
    f = np.zeros((s, s))
    u = np.zeros((s, s))
    a = 2 * np.random.rand(K, K) - 1
    r = 0.5  # r corresponds to the regularity of the functions, r=1 would be regular solution of Poisson
    
    # First, compute u
    for x in range(s):
        for y in range(s):
            u[x, y] = calculate_u(D * x / s, D * y / s, K, a, r)
            
    # Then, compute f based on u
    for x in range(s):
        for y in range(s):
            f[x, y] = calculate_f_helmholtz(D * x / s, D * y / s, K, a, r, omega, u[x, y])
            
    return f, u

print(f'{file_data}')

print('Begin File')
with h5py.File(file_data, 'w') as hf:
    if which_example == 'Poisson':
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
    
    elif which_example == 'Helmholtz':
        all_outputs = []
        for index in range(training_samples):
            omega = 5 * np.pi / 2  # As found in the benchmark
            inputs, outputs = evaluate_function_helmholtz(s, K, omega)
            
            all_outputs.append(outputs)

            if index % 50 == 0:
                print(index)
            
            hf.create_dataset(f'Sample_{index}/input', data=inputs, dtype=np.float32)
            hf.create_dataset(f'Sample_{index}/output', data=outputs, dtype=np.float32)

        all_outputs = np.array(all_outputs)

        hf.create_dataset('mean_data', data=1.0, dtype=np.float32)
        hf.create_dataset('mean_model', data=np.mean(all_outputs), dtype=np.float32)
        hf.create_dataset('std_model', data=np.std(all_outputs), dtype=np.float32)
    else:
        raise ValueError("which_example must be 'Poisson' or 'Helmholtz'")

    hf.create_dataset('K', data=K, dtype=np.float32)
    hf.create_dataset('Domain', data=D, dtype=np.float32)

if dist=='_out':
    Name_of_data_in=f'{which_example}Data_PDE' 
    file_data_in=os.path.join(path_to_Data,Name_of_data_in+'Domain'+str(D)+'.h5')
    if not os.path.exists(file_data_in): 
        raise FileNotFoundError(f"The file '{file_data_in}' must exist! Create first in distrubtion/trainings data")
    reader = h5py.File(file_data_in, 'r')
    
    file_data_out = file_data
    with h5py.File(file_data_out, 'a') as hf:
        if which_example == 'Poisson':
            min_data  = reader['min_inp'][()]
            max_data  = reader['max_inp'][()]
            min_model = reader['min_out'][()]
            max_model = reader['max_out'][()]
            hf['min_inp'][...] = min_data
            hf['max_inp'][...] = max_data
            hf['min_out'][...] = min_model
            hf['max_out'][...] = max_model
        elif which_example == 'Helmholtz':
            hf['mean_data'][...] = reader['mean_data'][()]
            hf['mean_model'][...] = reader['mean_model'][()]
            hf['std_model'][...] = reader['std_model'][()]


print('Finish File')