from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', type=str, default=config.logging_root)
p.add_argument('--root_dir', type=str, required=True)
p.add_argument('--batch_size', type=int, default=64)

opt = p.parse_args()

# Load dataset
dataset = datasets.ImageFolder(opt.root_dir)
num_classes = len(dataset.find_classes(opt.root_dir))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda_is_available() else "cpu")

# Load model
model = models.resnet50(pretrained=True)
# Replace last layer
model.fc = nn.Linear(512, num_classes)

