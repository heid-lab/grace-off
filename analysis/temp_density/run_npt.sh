#!/bin/bash

model_type='grace' # 'grace' or 'mace'
default_dtype='float32' # 'float32' or 'float64'
layer='2'
dataset='b_off'
sol="wat"
model_sizes=("large")

for model_size in "${model_sizes[@]}"; do
  for temp in 280 290 300 310 320 330; do

    job_name="job_${model_type}_${model_size}_${temp}.slurm"

    cat > $job_name <<EOF
#!/bin/bash
#SBATCH -p ADA,4090
#SBATCH --job-name=${model_type}_${model_size}_${temp}
#SBATCH --output=out_${model_type}_${model_size}_${temp}.txt

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-fennel_2

echo "Node(s): \$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

python run_sim.py \
  --model_size $model_size \
  --model_type $model_type \
  --sol $sol \
  --default_dtype $default_dtype \
  --layer $layer \
  --dataset $dataset \
  --temp $temp
EOF

    echo "Submitting $job_name"
    sbatch $job_name

  done
done
