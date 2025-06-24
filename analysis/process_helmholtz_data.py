'''
Created by: L. Kellijs
Date: 24/06/2025

This script is used to analyze Helmholtz data obtained from https://huggingface.co/datasets/camlab-ethz/Helmholtz
and to calculate the mean and standard deviation of the 'u' field across the samples.
'''


import h5py
import numpy as np
import os
from tqdm import tqdm

def analyze_helmholtz_data(file_path):
    """
    Analyzes the Helmholtz HDF5 dataset.

    This function performs the following steps:
    1. Checks if the file exists and is not a Git LFS pointer.
    2. Opens the HDF5 file and prints its top-level structure.
    3. Inspects a single sample to show the structure of 'a', 'u', and 'bc'.
    4. Calculates the global mean and standard deviation of the 'u' field
       across the training samples.
    5. Prints the calculated statistics.

    Args:
        file_path (str): The path to the Helmholtz HDF5 file.
    """
    # 1. Check if the file exists and is not a pointer file
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Please make sure you have downloaded the dataset and provided the correct path.")
        return

    # A simple check for a git-lfs pointer file.
    if os.path.getsize(file_path) < 1024:
        print(f"Warning: The file at '{file_path}' is very small.")
        print("It might be a Git LFS pointer file instead of the actual data.")
        print("Please ensure you have downloaded the full dataset using 'huggingface-cli' or another method.")
        with open(file_path, 'r') as f:
            print("File content:")
            print(f.read())
        return

    print(f"Analyzing Helmholtz data from: {file_path}")

    with h5py.File(file_path, 'r') as reader:
        # 2. Explore the structure
        sample_keys = list(reader.keys())
        num_samples = len(sample_keys)
        print(f"Found {num_samples} samples in the file (e.g., {sample_keys[0]}).")

        # 3. Inspect a single sample
        sample_name = sample_keys[0]
        print(f"\n--- Inspecting sample: {sample_name} ---")
        sample_group = reader[sample_name]
        print(f"Keys in sample: {list(sample_group.keys())}")

        # ! possibly wrong
        a_shape = sample_group['a'].shape
        u_shape = sample_group['u'].shape
        bc_value = sample_group['bc'][()]

        print(f"  - 'a' shape: {a_shape}, dtype: {sample_group['a'].dtype}")
        print(f"  - 'u' shape: {u_shape}, dtype: {sample_group['u'].dtype}")
        print(f"  - 'bc' value: {bc_value}, dtype: {sample_group['bc'].dtype}")
        print("--------------------------------------\n")

        # 4. Calculate mean and std for 'u' across training samples
        # According to Hugging Face, there are 19035 training samples.
        num_train_samples = 19035
        if num_samples < num_train_samples:
            print(f"Warning: Expected {num_train_samples} training samples, but found {num_samples} total samples.")
            print("Using all available samples for statistics.")
            num_train_samples = num_samples

        print(f"Calculating mean and std for 'u' across the first {num_train_samples} training samples...")

        # We use a simple sum and sum_of_squares approach which is efficient
        # if we process the data sample by sample.
        sum_u = np.zeros(1, dtype=np.float64)
        sum_sq_u = np.zeros(1, dtype=np.float64)
        total_pixels = 0
        
        for i in tqdm(range(num_train_samples)):
            sample_name = f'Sample_{i}'
            if sample_name in reader:
                u_data = reader[sample_name]['u'][:]
                sum_u += np.sum(u_data)
                sum_sq_u += np.sum(u_data**2)
                total_pixels += u_data.size
            else:
                print(f"Warning: Sample {sample_name} not found. Stopping calculation.")
                break

        if total_pixels > 0:
            # Calculate mean and std
            mean_u = sum_u / total_pixels
            std_u = np.sqrt(sum_sq_u / total_pixels - mean_u**2)

            # 5. Print the results
            print("\n--- Calculated Statistics for 'u' field ---")
            print(f"Mean: {mean_u[0]}")
            print(f"Standard Deviation: {std_u[0]}")
            print("-------------------------------------------")
            print("\nYou can now use these values in your HelmholtzDataset class.")

if __name__ == "__main__":
    # Path to your downloaded HDF5 file.
    # IMPORTANT: Update this path if you store the dataset elsewhere.
    h5_file_path = '/cluster/home/lkellijs/camlab-pino/data/Helmholtz/Helmholtz.h5'
    analyze_helmholtz_data(h5_file_path) 