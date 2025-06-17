#!/bin/bash
#SBATCH --output=Out/CNO%j.out
#SBATCH --time=2:00:00
#SBATCH -n 16
#SBATCH -G 1
#SBATCH --mem-per-cpu=512


source /cluster/home/harno/conda/etc/profile.d/conda.sh
conda activate env1
module load gcc/8.2.0 
module load python_gpu/3.11.2
module load ninja 
module load cuda/11.7.0

#python3 -u TrainPhysic_informed_CNO.py "$@"
python3 -u TrainCNO.py "$@"
