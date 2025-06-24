#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=eval-finite-laplacian
#SBATCH --output=logs/eval-finite-laplacian_%j.out
#SBATCH --error=logs/eval-finite-laplacian_%j.err


# load modules
module load stack/2024-06  gcc/12.2.0 python/3.12.8 cuda/12.8.0 

# Activate your environment
source /cluster/home/lkellijs/pino/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU information for verification
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run training script
python3 -u evaluate_finite_difference_laplacian.py