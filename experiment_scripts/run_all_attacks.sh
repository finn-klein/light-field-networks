#!/bin/bash -l

# Runs adversarial robustness attacks on LFN and FF-baseline.
# Saves results in folder hierarchy as follows:
# LFN/FF
#   -> Class
#       -> Attack type
#           -> type_class_attack.txt which contains epsilon/accuracy pairs
#
# Usage: bash run_all_attacks.sh log_root
# log_root should be the root folder path for the logs and should not exist
# (Model paths, data root hard-coded for now)

#SBATCH --job-name=lfn_2_l2gauss
#SBATCH --time=24:00:00
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

echo "experiment folder: $1"

# Create file structure
if [ -d "$1" ]; then
    echo "Directory $1 exists! Aborting"
    exit 1
fi
mkdir $1
cd $1

# --- LFN ---
mkdir LFN
cd LFN
echo "LFN"

for class in {0..12}; do
  mkdir $class
  cd $class
  for attack in "${attacks[@]}"
  do
    mkdir $attack
    cd $attack
    echo "$attack"
    python $script_path/adv_attacks.py --data_root $lfn_root_path --attack_name $attack --single_class_string $class --out_file lfn_${attack}_${class}.txt --checkpoint_path $lfn_ckpt_path
    cd ..
  done
  cd ..
done
cd ..

# --- FF ---
mkdir FF
cd FF
echo "FF"

for class in {0..12}; do
  mkdir $class
  cd $class
  for attack in "${attacks[@]}"
  do
    mkdir $attack
    cd $attack
    echo "$attack"
    python $script_path/adv_attack_resnet.py --data_root $ff_root_path --attack_name $attack --single_class_string $class --out_file ff_${attack}_${class}.txt --checkpoint_path $ff_ckpt_path
    cd ..
  done
  cd ..
done

exit 1