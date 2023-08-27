#!/bin/bash -l

#SBATCH --job-name=gradient_attacks_resnet
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

for class in {0..12}; do
  python $script_path/gradient_attack_resnet.py --data_root $ff_root_path --single_class_string ${class} --out_file $1/${class}.txt --checkpoint_path $ff_ckpt_path --n_steps 50
done

exit 1