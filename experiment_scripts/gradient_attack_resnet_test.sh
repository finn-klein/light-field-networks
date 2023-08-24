#!/bin/bash -l
#SBATCH --job-name=gradient_attack_resnet_test
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1 --partition=a100
#SBATCH --export=NONE

module load python
module load git

conda activate lf

attacks=("l2gauss" "l2uniform" "l2clippinggauss" "l2clippinguniform" "linfuniform") # "l2repeatedgauss" "l2repeateduniform" "l2clippingrepeatedgauss" "l2clippingrepeateduniform" "linfrepeateduniform")
script_path="/home/woody/iwi9/iwi9015h/light-field-networks/experiment_scripts"
lfn_root_path="/home/woody/iwi9/iwi9015h/nmr/NMR_Dataset"
lfn_ckpt_path="/home/vault/iwi9/iwi9015h/experiments/train00/nmr/64_128_None/checkpoints/model_epoch_0006_iter_040000.pth"
ff_root_path="/home/woody/iwi9/iwi9015h/nmr_ff_test"
ff_ckpt_path="/home/vault/iwi9/iwi9015h/experiments/train_baseline00/checkpoints/best.pth"

python $script_path/gradient_attack_resnet.py --data_root $ff_root_path --single_class_string 0 --out_file $WORK/experiments/grad_resnet_test/test2.txt --checkpoint_path $ff_ckpt_path --n_steps 50

exit 1