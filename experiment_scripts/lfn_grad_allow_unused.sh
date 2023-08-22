#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=lfn_gradient_attacks
#SBATCH --export=None

module load python
module load cuda/11.3.1
module load git
source activate lf

mkdir adv_attacks/$SLURM_JOB_ID

#for y in {0..12}; do
python experiment_scripts/lfn_grad_allow_unused.py --data_root /home/woody/iwi9/iwi9015h/nmr/NMR_Dataset \
--checkpoint_path /home/vault/iwi9/iwi9015h/experiments/train00/nmr/64_128_None/checkpoints/model_epoch_0006_iter_040000.pth \
--num_inference_iters 500 \
--num_instances_per_class 256 \
--batch_size 256 \
--lr 5e-4 \
--single_class_string 0 \
--adv_epsilon 0.001 \
--out_folder adv_attacks/$SLURM_JOB_ID
#done
