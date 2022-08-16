#!/bin/bash -l
#
#SBATCH --time=09:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=SRN_attacks_class_0
#SBATCH --export=None

module load python
module load cuda/11.3.1
module load git
source activate srns

python adv_attacks.py \
--data_root /home/woody/iwi9/iwi9015h/nmr/NMR_Dataset \
--checkpoint_path /home/vault/iwi9/iwi9015h/SRNs/train_srn_from_ckpt/checkpoints/epoch_0001_iter_028000.pth \
--set test \
--batch_size 64 \
--single_class_string 0 \
--num_instances_per_class 16