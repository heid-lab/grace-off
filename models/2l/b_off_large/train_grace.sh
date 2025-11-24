#!/bin/bash

#SBATCH --partition=GPU-a100  # select a partition
#SBATCH --nodes=1  # select number of nodes (max 2, typically 1)
#SBATCH --ntasks-per-node=12  # select number of tasks per node (essentially CPU cores)
#SBATCH --time=7-00:00:00  # request 7 days of max runtime
#SBATCH --gres=gpu:a100:3

echo "Node(s): $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

sleep 120

source /home/johannes.karwounopoulos/micromamba/etc/profile.d/mamba.sh
micromamba activate /home/johannes.karwounopoulos/micromamba/envs/grace-linus

export TF_USE_LEGACY_KERAS=1

gracemaker -r -m

gracemaker -r -s
