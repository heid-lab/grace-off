#!/bin/bash

#SBATCH --partition=GPU-l40s       # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:1                # Use GPU
#SBATCH --nodes=1                   # select number of nodes
#SBATCH --ntasks-per-node=12        # select number of tasks per node
#SBATCH --time=7-00:00:00           # Maximum time

# Load any required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel

model_size=$1
model_type=$2
default_dtype=$3
layer=$4
dataset=$5

sol="wat"

echo "Running NPT with model_size=$model_size, model_type=$model_type, default_dtype=$default_dtype, layer=$layer, dataset=$dataset"
python ase_npt.py --model_size $model_size --model_type $model_type --sol $sol --default_dtype $default_dtype --layer $layer --dataset $dataset


# python ase_npt.py --model_size small --model_type mace --sol $sol --default_dtype float32

# for using a GRACE model trained on a_wpS dataset
# python ase_npt.py --model_size small --model_type grace --sol $sol --dataset a_wpS --layer 1 --default_dtype float32