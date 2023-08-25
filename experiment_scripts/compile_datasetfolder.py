"""
Compile the NMR ShapeNet dataset into a torchvision ImageFolder dataset.
"""

from PIL import Image
import configargparse
import sys, os
sys.path.append("/home/woody/iwi9/iwi9015h/light-field-networks")
import multiclass_dataio

p = configargparse.ArgumentParser()
p.add("-c", "--config_filepath", required=False, is_config_file=True)
p.add_argument("--data_root", required=True)
p.add_argument("--out_path", required=True)
p.add_argument("--angle", type=int, required=True)
p.add_argument("--set", required=True)

opt = p.parse_args()

classes = list(multiclass_dataio.string2class_dict.keys())

for c in classes:
    if not os.path.exists(f"{opt.out_path}/{c}"):
        os.mkdir(f"{opt.out_path}/{c}")
    
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
        img_raw = x[opt.angle]['rgb'].numpy().reshape(64, 64, 3) # range (-1, 1)
        img_raw /= 2 # (-.5, .5)
        img_raw += .5 # (0, 1)
        img_raw *= 255 # (0, 255)
        img = Image.fromarray(img_raw.astype("uint8"), mode="RGB")
        img.save(f"{opt.out_path}/{c}/{x.instance_name}.png")
