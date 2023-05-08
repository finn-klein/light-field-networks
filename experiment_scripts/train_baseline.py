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
from torch.utils.tensorboard import SummaryWriter

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', type=str, required=True)
p.add_argument('--root_dir_train', type=str, required=True)
p.add_argument('--root_dir_val', type=str, required=False)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--num_epochs', type=int, default=10000)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--epochs_til_ckpt', type=int, default=5)

opt = p.parse_args()

dsets = dict()
dloaders = dict()

# helper function to get numpy array from PIL image
def collate(data):
    d = [(np.asarray(sample), idx) for sample, idx in data]
    d = list(zip(*d))
    return d[0], d[1]

# Load datasets
dsets['train'] = datasets.ImageFolder(opt.root_dir_train)
num_classes = len(dsets['train'].find_classes(opt.root_dir_train))
dloaders['train'] = torch.utils.data.DataLoader(dsets['train'], batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=collate)

if opt.root_dir_val is not None:
    dsets['val'] = datasets.ImageFolder(opt.root_dir_val)
    num_classes = len(dsets['val'].find_classes(opt.root_dir_val))
    dloaders['val'] = torch.utils.data.DataLoader(dsets['val'], batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=collate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create logging directories
summaries_path = f"{opt.logging_root}/summaries"
if not os.path.exists(summaries_path):
    os.mkdir(summaries_path)

checkpoints_path = f"{opt.logging_root}/checkpoints"
if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)

writer = SummaryWriter(summaries_path, flush_secs=10)

# Load model
model = models.resnet50(pretrained=True)
# Replace last layer
model.fc = nn.Linear(512, num_classes)

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.train()

    total_steps = 0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                if opt.root_dir_val == None:
                    continue
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # stats
                single_loss = loss.item() * inputs.size(0)
                running_loss += single_loss
                writer.add_scalar(f"loss/{phase}", single_loss, total_steps)
                running_corrects += torch.sum(preds == labels.data)

                total_steps += 1
                if total_steps % opt.iters_til_ckpt == 0:
                    # save model
                    torch.save(model.state_dict(), f"{checkpoints_path}/model_epoch_{epoch}_iter_{total_steps}.pth")
            
            # epoch stats
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            writer.add_scalar(f"loss/epoch_{phase}", epoch_loss, epoch * len(dataloaders[phase]))
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            writer.add_scalar(f"acc/epoch_{phase}", epoch_acc, epoch*len(dataloaders[phase]))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        if epoch != 0 and epoch % opt.epochs_til_ckpt == 0:
            torch.save(model.state_dict(), f"{checkpoints_path}/model_epoch_{epoch}_iter_{total_steps}.pth")
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# initialize optimizer
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_model(model, dloaders, optimizer, criterion, opt.num_epochs)