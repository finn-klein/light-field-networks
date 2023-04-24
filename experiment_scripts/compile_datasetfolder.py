"""
Compile the NMR ShapeNet model into a torchvision ImageFolder model.
"""

import torch, torchvision
import os
import PIL
import configargparse

p = configargparse.ArgumentParser()
p.add("-c", "--config_filepath", required=False, is_config_file=True)
p.add_argument("--data_root", required=True)
p.add_argument("--out_path", required=True)
p.add_argument("--angle", required=True)
p.add_argument("--upscale", type=int, default=224)