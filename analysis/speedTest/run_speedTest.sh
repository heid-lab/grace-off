#!/bin/bash
#SBATCH --partition=GPU-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=7-00:00:00
#SBATCH --job-name=speedtest

export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh

mkdir -p $SCRATCH/tf_tmp 2>/dev/null || mkdir -p $HOME/tf_tmp
export TMPDIR=${SCRATCH:-$HOME}/tf_tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR

conda activate ai-fennel_cuda12
python -u speedTest.py --arch grace

conda activate openmm
python -u speedTest.py --arch mace

conda activate fairchem
python -u speedTest.py --arch uma
