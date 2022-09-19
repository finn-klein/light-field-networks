#!/bin/bash -l
#
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=lfn_noise_attacks
#SBATCH --export=None

module load python
module load cuda/11.3.1
module load git
source activate lf


for x in l2gauss l2uniform l2clippinggauss l2clippinguniform linfuniform l2repeatedgauss l2repeateduniform l2clippingrepeatedgauss l2clippingrepeateduniform linfrepeateduniform; do
    for y in {0..12}; do
	python experiment_scripts/lfn_noise_attacks.py --data_root /home/woody/iwi9/iwi9015h/nmr/NMR_Dataset \
	--checkpoint_path /home/vault/iwi9/iwi9015h/light_fields/rec_from_scratch/64_64_None/checkpoints/model_current.pth \
	--num_inference_iters 1000 \
	--num_instances_per_class 256 \
	--batch_size 256 \
	--lr 1e-4 \
	--attack_name $x \
	--single_class_string $y > noise_attacks/class-$y-attack-$x.txt
    done
done
