#!/bin/bash -l

#SBATCH --job-name=lfn_2_l2gauss
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1 --partition=a100
#SBATCH --export=NONE

module load python
module load git

conda activate lf

out_path="/home/woody/iwi9/iwi9015h/experiments/batchtest"
script_path="/home/woody/iwi9/iwi9015h/light-field-networks/experiment_scripts"
lfn_root_path="/home/woody/iwi9/iwi9015h/nmr/NMR_Dataset"
lfn_ckpt_path="/home/vault/iwi9/iwi9015h/experiments/train00/nmr/64_128_None/checkpoints/model_epoch_0006_iter_040000.pth"
ff_root_path="/home/woody/iwi9/iwi9015h/nmr_ff_test"
ff_ckpt_path="/home/vault/iwi9/iwi9015h/experiments/train_baseline00/checkpoints/best.pth"

cd /home/woody/iwi9/iwi9015h/light-field-networks/
python experiment_scripts/adv_attacks.py --data_root $lfn_root_path --checkpoint_path $lfn_ckpt_path --attack_name l2gauss \
 --single_class_string 2 --out_file ${out_path}/lfn_2_l2gauss.txt