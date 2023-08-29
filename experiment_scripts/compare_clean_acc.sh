#!/bin/bash -l

#SBATCH --job-name=compare_clean_acc
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1 --partition=a100
#SBATCH --export=NONE

module load python
module load git

conda activate lf

script_path="/home/woody/iwi9/iwi9015h/light-field-networks/experiment_scripts"
lfn_root_path="/home/woody/iwi9/iwi9015h/nmr/NMR_Dataset"
lfn_ckpt_path="/home/vault/iwi9/iwi9015h/experiments/train00/nmr/64_128_None/checkpoints/model_epoch_0006_iter_040000.pth"
ff_root_path="/home/woody/iwi9/iwi9015h/nmr_ff_test_2"
ff_ckpt_path="/home/vault/iwi9/iwi9015h/experiments/train_baseline01/checkpoints/best.pth"

mkdir $1
python $script_path/clean_accuracy.py --data_root_ff $ff_root_path --data_root_lfn $lfn_root_path \
    --checkpoint_path_ff $ff_ckpt_path --checkpoint_path_lfn $lfn_ckpt_path \
    --out_path $1
exit 1