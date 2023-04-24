"""
Compile the NMR ShapeNet dataset into a torchvision ImageFolder dataset.
"""

import torch, torchvision
import os
import PIL
import configargparse
import sys, os
sys.path.append("/home/woody/iwi9/iwi9015h/light-field-networks")

p = configargparse.ArgumentParser()
p.add("-c", "--config_filepath", required=False, is_config_file=True)
p.add_argument("--data_root", required=True)
p.add_argument("--out_path", required=True)
p.add_argument("--angle", type=int, required=True)
p.add_argument("--set", required=True)

opt = p.parse_args()

classes = list(multiclass_dataio.string2class_dict.keys())

for c in classes:
    if not os.path.exists(f"{out_path}/{c}"):
        os.mkdir(f"{out_path}/{c}")
    
    dataset = multiclass_dataio.SceneClassDataset(num_context=1,
                                        num_trgt=1,
                                        root_dir=opt.data_root,
                                        query_sparsity=None,
                                        img_sidelength=64,
                                        vary_context_number=False,
                                        cache=None,
                                        dataset_type=opt.set,
                                        specific_classes=[c])
    
    for x in dataset.all_instances:
        img = Image.fromarray((x[opt.angle]['rgb'].numpy().reshape(64, 64, 3)*255).astype("uint8"), mode="RGB")
        img.save(f"{out_path}/{c}/{x.instance_name}.png")
