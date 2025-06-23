#!/usr/bin/env python3
"""
Evaluate the validity of the Fourier-based Laplacian method for the Poisson equation.

This script loads the training set and for each input-label pair:
1. Computes the Fourier-based Laplacian of the label using the method from ModulePDELoss.py
2. Takes its negative
3. Compares it to the input (which should be -∇²u for the Poisson equation)
4. Calculates various error metrics

Results are stored in a formatted text file.
"""

import sys
import os
import torch
import numpy as np
import json
from datetime import datetime

# Configuration flags - Set these at the top of the script
DEVICE = 'cpu'  # Change to 'cuda' if GPU available
BATCH_SIZE = 16
TRAINING_SAMPLES = 1024

# Add parent directory to path to import project modules
if '..' not in sys.path:
    sys.path.append('..')

from Problems.CNOBenchmarks import SinFrequency as CNO_SinFrequency
from Problems.FNOBenchmarks import SinFrequency as FNO_SinFrequency
from Physics_NO.loss_functions.ModulePDELoss import Unnormalize, Laplace

def detect_model_type(model_path):
    """Detect whether the model is CNO or FNO based on the path or architecture file."""
    if 'CNO' in model_path or 'cno' in model_path:
        return 'CNO'
    elif 'FNO' in model_path or 'fno' in model_path:
        return 'FNO'
    else:
        # Check architecture file if path doesn't contain type info
        arch_path = os.path.join(model_path, 'net_architecture.txt')
        if os.path.exists(arch_path):
            import pandas as pd
            df = pd.read_csv(arch_path, header=None, index_col=0)
            arch_dict = df.to_dict()[1]
            if 'in_size' in arch_dict:
                return 'CNO'
            elif 'width' in arch_dict:
                return 'FNO'
        
        # Default fallback
        print("Warning: Could not detect model type from path. Assuming CNO.")
        return 'CNO'

def convert_fno_to_cno_format(x):
    """Convert FNO format (batch, height, width, channel) to CNO format (batch, channel, height, width)"""
    return x.permute(0, 3, 1, 2)

def load_dataset(model_path, device, batch_size, training_samples):
    """Load the dataset using appropriate class based on model type."""
    # Load model architecture from the pretrained model's directory
    import pandas as pd
    net_architecture_path = os.path.join(model_path, 'net_architecture.txt')
    df = pd.read_csv(net_architecture_path, header=None, index_col=0)
    model_architecture_ = df.to_dict()[1]
    model_architecture_ = {key: int(value) if str(value).isdigit() else float(value) if '.' in str(value) else value for key, value in df.to_dict()[1].items()}

    model_type = detect_model_type(model_path)
    
    # Get the appropriate size parameter based on model type
    if model_type == 'CNO':
        in_size = model_architecture_["in_size"]
    elif model_type == 'FNO':
        in_size = model_architecture_["width"]  # FNO uses 'width' instead of 'in_size'

    # Load data using appropriate class
    if model_type == 'CNO':
        example = CNO_SinFrequency(model_architecture_, device, batch_size=batch_size, training_samples=training_samples)
    elif model_type == 'FNO':
        example = FNO_SinFrequency(model_architecture_, device, batch_size=batch_size, training_samples=training_samples)

    train_loader = example.train_loader
    normalization_values = train_loader.dataset.get_max_and_min()
    
    return train_loader, normalization_values, model_type, in_size

def main():
    # Default model path - modify as needed
    model_path = "../TrainedModels/CNO_1024poisson"
    
    print(f"Evaluating Fourier-based Laplacian Method")
    print(f"Device: {DEVICE}")
    print(f"Model Path: {model_path}")
    print("-" * 60)
    
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
    
    # Load dataset
    train_loader, normalization_values, model_type, in_size = load_dataset(
        model_path, device, BATCH_SIZE, TRAINING_SAMPLES
    )
    
    print(f"Model type: {model_type}")
    print(f"Grid size: {in_size}")
    print(f"Normalization values: {normalization_values}")
    
    # Initialize Fourier-based Laplacian
    fourier_laplace = Laplace(s=in_size, D=1.0)
    fourier_laplace = fourier_laplace.to(device)
    
    # Initialize unnormalization function
    unnormalize_fn = Unnormalize('poisson', normalization_values)
    
    # Storage for results
    all_relative_errors = []
    all_max_errors = []
    all_min_errors = []
    all_std_errors = []
    
    print(f"Processing {len(train_loader)} batches...")
    
    batch_count = 0
    for input_batch, label_batch in train_loader:
        batch_count += 1
        print(f"Processing batch {batch_count}/{len(train_loader)}")
        
        # Convert FNO format to CNO format if needed
        if model_type == 'FNO':
            input_batch = convert_fno_to_cno_format(input_batch)
            label_batch = convert_fno_to_cno_format(label_batch)
        
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Unnormalize the data
        input_unnorm, label_unnorm = unnormalize_fn(input=input_batch, output=label_batch)
        
        # Process each sample in the batch
        for i in range(input_batch.shape[0]):
            # Get individual samples
            input_sample = input_unnorm[i]  # Shape: [1, H, W]
            label_sample = label_unnorm[i]  # Shape: [1, H, W]
            
            # Compute Fourier-based Laplacian of the label
            # The Fourier Laplacian expects input without channel dimension for single samples
            # So we squeeze the channel dimension
            label_squeezed = label_sample.squeeze(0)  # Shape: [H, W]
            fourier_laplacian = fourier_laplace(label_squeezed)  # Shape: [H, W]
            
            # Take negative of Laplacian
            neg_fourier_laplacian = -fourier_laplacian
            
            # Input sample for comparison (squeeze channel dimension)
            input_squeezed = input_sample.squeeze(0)  # Shape: [H, W]
            
            # Convert to numpy for calculations
            input_np = input_squeezed.cpu().numpy()
            neg_laplacian_np = neg_fourier_laplacian.cpu().numpy()
            
            # Calculate error
            error = np.abs(input_np - neg_laplacian_np)
            
            # Calculate relative error (L2 norm)
            rel_error = np.linalg.norm(error) / (np.linalg.norm(input_np) + 1e-12)
            
            # Calculate max, min, and std of errors over pixel values
            max_error = np.max(error)
            min_error = np.min(error)
            std_error = np.std(error)
            
            # Store results
            all_relative_errors.append(rel_error)
            all_max_errors.append(max_error)
            all_min_errors.append(min_error)
            all_std_errors.append(std_error)
    
    # Calculate summary statistics
    avg_rel_error = np.mean(all_relative_errors)
    avg_max_error = np.mean(all_max_errors)
    avg_min_error = np.mean(all_min_errors)
    avg_std_error = np.mean(all_std_errors)
    
    std_rel_error = np.std(all_relative_errors)
    std_max_error = np.std(all_max_errors)
    std_min_error = np.std(all_min_errors)
    std_std_error = np.std(all_std_errors)
    
    # Create results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'laplacian_method': 'Fourier-based (from ModulePDELoss.py)',
            'device': str(device),
            'model_path': model_path,
            'model_type': model_type,
            'grid_size': in_size,
            'training_samples': TRAINING_SAMPLES,
            'batch_size': BATCH_SIZE,
            'total_samples_processed': len(all_relative_errors)
        },
        'normalization_values': normalization_values,
        'statistics': {
            'average_relative_error': {
                'mean': float(avg_rel_error),
                'std': float(std_rel_error)
            },
            'average_maximum_error': {
                'mean': float(avg_max_error),
                'std': float(std_max_error)
            },
            'average_minimum_error': {
                'mean': float(avg_min_error),
                'std': float(std_min_error)
            },
            'average_standard_deviation_error': {
                'mean': float(avg_std_error),
                'std': float(std_std_error)
            }
        },
        'raw_data': {
            'relative_errors': [float(x) for x in all_relative_errors],
            'max_errors': [float(x) for x in all_max_errors],
            'min_errors': [float(x) for x in all_min_errors],
            'std_errors': [float(x) for x in all_std_errors]
        }
    }
    
    # Save results to file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"fourier_laplacian_evaluation_{timestamp_str}.txt"
    
    with open(output_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FOURIER-BASED LAPLACIAN EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Laplacian Method: Fourier-based (from ModulePDELoss.py)\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Grid Size: {in_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Samples Processed: {len(all_relative_errors)}\n\n")
        
        f.write("NORMALIZATION VALUES:\n")
        for key, value in normalization_values.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average Relative Error:    {avg_rel_error:.6e} ± {std_rel_error:.6e}\n")
        f.write(f"Average Maximum Error:     {avg_max_error:.6e} ± {std_max_error:.6e}\n")
        f.write(f"Average Minimum Error:     {avg_min_error:.6e} ± {std_min_error:.6e}\n")
        f.write(f"Average Std Deviation:     {avg_std_error:.6e} ± {std_std_error:.6e}\n\n")
        
        f.write("DETAILED EXPLANATION:\n")
        f.write("-" * 50 + "\n")
        f.write("For the Poisson equation -∇²u = f, we expect:\n")
        f.write("  Input (f) ≈ -∇²(Label)\n\n")
        f.write("The relative error is calculated as:\n")
        f.write("  ||f - (-∇²u)||₂ / ||f||₂\n\n")
        f.write("This evaluation uses the Fourier-based Laplacian implementation\n")
        f.write("from the ModulePDELoss.py file, which uses FFT to compute derivatives.\n\n")
        f.write("Lower relative errors indicate better agreement between\n")
        f.write("the Fourier method and the expected physics.\n\n")
        
        # Also save as JSON for easier programmatic access
        json_filename = output_filename.replace('.txt', '.json')
        with open(json_filename, 'w') as json_f:
            json.dump(results, json_f, indent=2)
    
    print(f"\nEvaluation Complete!")
    print(f"Results saved to: {output_filename}")
    print(f"JSON data saved to: {json_filename}")
    print(f"\nSummary:")
    print(f"  Average Relative Error: {avg_rel_error:.6e} ± {std_rel_error:.6e}")
    print(f"  Average Maximum Error:  {avg_max_error:.6e} ± {std_max_error:.6e}")
    print(f"  Average Minimum Error:  {avg_min_error:.6e} ± {std_min_error:.6e}")
    print(f"  Average Std Deviation:  {avg_std_error:.6e} ± {std_std_error:.6e}")

if __name__ == "__main__":
    main() 