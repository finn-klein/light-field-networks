# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

sys.path.append("../foolbox")
sys.path.append("../eagerpy")
import foolbox as fb
import configargparse
import os, time, datetime
import config

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
from models import LFAutoDecoder
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)
p.add("--checkpoint_path", required=True)
p.add("--data_root", required=True)
p.add("--batch_size", default=64)
opt = p.parse_args()

# helper function to get numpy array from PIL image
def collate(data):
    d = list(zip(*data))
    imgs = d[0]
    labels = d[1]
    
    # cast PIL.Image into np.array, normalize to [0; 1]
    imgs = [np.asarray(img, dtype="float32") / 255 for img in imgs]
    # convert list of imgs to np.array before wrapping in tensor (performance reasons)
    imgs = np.array(imgs)
    imgs = torch.tensor(imgs)
    # permute dimensions as expected by ResNet
    imgs = torch.permute(imgs, (0, 3, 1, 2)) # channels, rgb, x, y
    
    labels = torch.tensor(labels)
    
    return imgs, labels


# Initialize dataset
dataset = datasets.ImageFolder(opt.data_root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
num_classes = len(dsets['train'].find_classes(opt.root_dir_train)[0])

model = models.resnet50().eval()
in_feat = model.fc.in_features
# modify last layer
model.fc = torch.nn.Linear(in_feat, num_classes)

state_dict = torch.load(opt.checkpoint_path)
model.load_state_dict(state_dict)
# preprocessing??
fmodel = fb.PyTorchModel(model, bounds=(0, 1))


robust_accs = list()
for imgs, labels in dataloader:
    print(f"clean accuracy:  {fb.accuracy(fmodel, imgs, labels) * 100:.1f} %")
    attack = fb.attacks.L2AdditiveGaussianNoiseAttack()

    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]

    epsilons = [x*256 for x in epsilons]
    print('labels', labels.size())
    raw_advs, clipped_advs, success = attack(fmodel, inputs=imgs, criterion=labels, epsilons=epsilons)
    robust_accuracy = 1 - success.float().mean(axis=-1)
    robust_accs.append(robust_accuracy)

    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
    print()
