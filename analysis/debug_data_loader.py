import h5py
import numpy as np
import os

def check_h5_file(file_path):
    """
    Checks if the HDF5 file contains the necessary datasets for ModulePDELoss.py.
    """
    print(f"--- Debugging Data Loader for: {file_path} ---")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    required_attributes = ['mean_data', 'mean_model', 'std_model', 'K', 'Domain']
    
    try:
        with h5py.File(file_path, 'r') as hf:
            print("File opened successfully.")
            
            # 1. Check for top-level attributes
            print("\nChecking for top-level attributes...")
            all_attributes_found = True
            for attr in required_attributes:
                if attr in hf:
                    value = hf[attr][()]
                    print(f"  - Found '{attr}': {value}")
                else:
                    print(f"  - Error: Required attribute '{attr}' not found.")
                    all_attributes_found = False
            
            if not all_attributes_found:
                 print("Error: Missing one or more required top-level attributes.")
                 return

            # 2. Check for sample data structure
            print("\nChecking for sample data structure (testing Sample_0)...")
            if 'Sample_0' in hf:
                print("  - Found group 'Sample_0'.")
                sample_group = hf['Sample_0']
                if 'input' in sample_group and 'output' in sample_group:
                    print("  - Found 'input' and 'output' datasets in 'Sample_0'.")
                    input_shape = sample_group['input'].shape
                    output_shape = sample_group['output'].shape
                    print(f"    - 'input' shape: {input_shape}")
                    print(f"    - 'output' shape: {output_shape}")
                else:
                    print("  - Error: 'input' or 'output' dataset missing in 'Sample_0'.")
            else:
                print("  - Error: 'Sample_0' not found.")

            print("\n--- Debugging Summary ---")
            if all_attributes_found and 'Sample_0' in hf and 'input' in hf['Sample_0'] and 'output' in hf['Sample_0']:
                print("Success! The data file seems to have the correct structure and attributes for ModulePDELoss.py.")
            else:
                print("Failure. The data file is missing required attributes or has an incorrect structure.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    # Based on Physics_NO/helper_functions/Create_Data.py
    which_example = 'Helmholtz'
    path_to_Data= '/cluster/home/lkellijs/data'
    Name_of_data=f'{which_example}Data_PDE'
    D=1
    file_data=os.path.join(path_to_Data,Name_of_data+'Domain'+str(D)+'.h5')
    
    check_h5_file(file_data) 