#!/bin/bash
#SBATCH -p 4090

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel_2

model_type='mace'
model_size='small'
dataset='a_wpS'
layers='2'
default_dtype='float64'



echo "Get water dimer energy"
python grace_dimer_energies.py --model_type $model_type --model_size $model_size --dataset $dataset --layers $layers --default_dtype $default_dtype

