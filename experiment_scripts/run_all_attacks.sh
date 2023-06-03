#!/bin/bash

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

attacks=("l2gauss" "l2uniform" "l2clippinggauss" "l2clippinguniform" "linfuniform" "l2repeatedgauss" "l2repeateduniform" "l2clippingrepeatedgauss" "l2clippingrepeateduniform" "linfrepeateduniform")



exit 1


# Create file structure
if [ -d "$1" ]; then
    echo "Directory $1 exists! Aborting"
    exit 1
fi
mkdir $1

# --- LFN ---
cd $1
mkdir LFN
cd LFN

for class in {0..12}; do
    mkdir $class
    cd $class
    for attack in "${attacks[@]}"
    do
      mkdir $attack
      echo "$attack"
      
    done
done