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
p.add("--batch_size", default=256)
p.add("--attack_name", required=True)
p.add("--out_file", type=str, required=False)
p.add("--single_class_string", type=str, required=True)
opt = p.parse_args()

if opt.out_file is not None:
    out_file = open(opt.out_file, "w")

attacks = {"l2gauss": fb.attacks.L2AdditiveGaussianNoiseAttack(),
           "l2uniform": fb.attacks.L2AdditiveUniformNoiseAttack(),
           "l2clippinggauss": fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack(),
           "l2clippinguniform": fb.attacks.L2ClippingAwareAdditiveUniformNoiseAttack(),
           "linfuniform": fb.attacks.LinfAdditiveUniformNoiseAttack(),
           "l2repeatedgauss": fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack(),
           "l2repeateduniform": fb.attacks.L2RepeatedAdditiveUniformNoiseAttack(),
           "l2clippingrepeatedgauss": fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(),
           "l2clippingrepeateduniform": fb.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(),
           "linfrepeateduniform": fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack()
           }

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

if opt.single_class_string is not None:
    single_class_dataset = [(item, label) for (item, label) in dataset if label==opt.single_class_string]
    single_class_dataloader = torch.utils.data.DataLoader(single_class_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
else:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
num_classes = len(dataset.find_classes(opt.data_root)[0])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50().eval()
model = model.to(device)
in_feat = model.fc.in_features
# modify last layer
model.fc = torch.nn.Linear(in_feat, num_classes)

state_dict = torch.load(opt.checkpoint_path)
model.load_state_dict(state_dict)
# preprocessing??
fmodel = fb.PyTorchModel(model, bounds=(0, 1))


robust_accs = list()
for imgs, labels in dataloader:
    imgs = imgs.to(device)
    labels = labels.to(device)
    attack = attacks[opt.attack_name]

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

    if opt.out_file is not None:
        for (eps, acc) in zip(epsilons, robust_accuracy):
            out_file.write(f"{eps:<6} {robust_accuracy.item() * 100:4.1f}\n")
    else:
        print("robust accuracy for perturbations with")
        for eps, acc in zip(epsilons, robust_accuracy):
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        print()
