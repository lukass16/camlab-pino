import h5py
import numpy as np

#Changing max min for data out of distriubtion

file_data='/cluster/scratch/harno/data/PoissonData_PDEDomain2.h5'
reader = h5py.File(file_data, 'r')
min_data  = reader['min_inp'][()]
max_data  = reader['max_inp'][()]
min_model = reader['min_out'][()]
max_model = reader['max_out'][()]


file_data_out = '/cluster/scratch/harno/data/PoissonData_PDE_out25Domain2.h5'
with h5py.File(file_data_out, 'a') as hf:
    hf['min_inp'][...] = min_data
    hf['max_inp'][...] = max_data
    hf['min_out'][...] = min_model
    hf['max_out'][...] = max_model

print("Data updated successfully.")

#Add domain
#with h5py.File(file_data, 'a') as hf:
#     hf.create_dataset('Domain', data=2, dtype=np.float32)


