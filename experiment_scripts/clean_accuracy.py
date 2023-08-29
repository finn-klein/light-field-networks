# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

sys.path.append("/home/woody/iwi9/iwi9015h/foolbox")
sys.path.append("/home/woody/iwi9/iwi9015h/eagerpy")
import foolbox as fb
import configargparse
import os, time, datetime
import config

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import multiclass_dataio
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from models import LFAutoDecoder
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--data_root_ff', required=True)
p.add_argument('--data_root_lfn', required=True)
p.add_argument('--checkpoint_path_lfn', type=str)
p.add_argument('--checkpoint_path_ff', type=str)

p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--img_sidelength', type=int, default=64, required=False)
p.add_argument('--batch_size', type=int, default=128)
p.add_argument('--num_inference_iters', type=int, default=150)
p.add_argument('--set', default='test')

p.add_argument('--specific_observation_idcs', type=str)
p.add_argument('--max_num_instances', type=int, default=128)
p.add_argument('--max_num_observations', type=int, default=128, required=False)
p.add_argument('--num_instances_per_class', type=int, default=128, required=False)
p.add_argument('--out_path', type=str, required=False)
opt = p.parse_args()

if opt.out_path is not None:
    out_file_lfn = open(f"{opt.out_path}/lfn.txt", "w")
    out_file_ff = open(f"{opt.out_path}/ff.txt", "w")

lr = opt.lr
num_iters = opt.num_inference_iters

if opt.specific_observation_idcs is not None:
    specific_observation_idcs = util.parse_comma_separated_integers(opt.specific_observation_idcs)
else:
    specific_observation_idcs = None

if opt.num_instances_per_class is not None:
    num_instances_per_class = opt.num_instances_per_class
else:
    num_instances_per_class = None

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

for c in range(13):
    print("Loading datasets")
    class_id = [multiclass_dataio.class2string_dict[c]]
    dataset_lfn = multiclass_dataio.SceneClassDataset(num_context=1, num_trgt=1,
                                                        root_dir=opt.data_root_lfn, query_sparsity=None,
                                                        img_sidelength=opt.img_sidelength, vary_context_number=True,
                                                        specific_observation_idcs=specific_observation_idcs, cache=None,
                                                        max_num_instances=opt.max_num_instances,
                                                        # max_num_observations_per_instance=opt.max_num_observations_train,
                                                        dataset_type=opt.set,
                                                        viewlist="/home/woody/iwi9/iwi9015h/light-field-networks/experiment_scripts/viewlists/src_dvr.txt",
                                                        num_instances_per_class=num_instances_per_class,
                                                        specific_classes=class_id)
    dataloader_lfn = DataLoader(dataset_lfn, batch_size=opt.batch_size, shuffle=False,
                            drop_last=True, num_workers=0)

    # Initialize dataset
    dataset_ff = datasets.ImageFolder(opt.data_root_ff)
    dataset_ff = [(item, label) for (item, label) in dataset_ff if label==c]
    dataloader_ff = DataLoader(dataset_ff, batch_size=opt.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    print("Initializing models")

    # --- init LFN ---

    model_lfn = LFAutoDecoder(latent_dim=256, num_instances=dataset_lfn.num_instances, classify=True).cuda()
    model_lfn.eval()

    if opt.checkpoint_path_lfn is not None:
        print(f"Loading weights from {opt.checkpoint_path_lfn}...")
        state_dict = torch.load(opt.checkpoint_path_lfn)
        del state_dict['latent_codes.weight']
        model_lfn.load_state_dict(state_dict, strict=False)

    fmodel_lfn = fb.PyTorchModel(model_lfn, bounds=(-1, 1))

    # --- init FF ---

    model_ff = models.resnet50().eval()
    model_ff = model_ff.to(device)
    in_feat = model_ff.fc.in_features
    model_ff.fc = torch.nn.Linear(in_feat, num_classes)
    state_dict = torch.load(opt.checkpoint_path_ff)
    model_ff.load_state_dict(state_dict)
    fmodel_ff = fb.PyTorchModel(model_ff, bounds=(0, 1))

    print("Evaluating")

    # --- eval LFN ---

    for model_input, ground_truth in iter(dataloader_lfn):
        inputs = model_input # (b, sidelength**2, 3)
        rgb = model_input['query']['rgb'].cuda()
        intrinsics = model_input['query']['intrinsics'].cuda()
        pose = model_input['query']['cam2world'].cuda()
        uv = model_input['query']['uv'].cuda().float()
        labels = model_input['query']['class'].squeeze().cuda() # (b)

        model_lfn.pose = pose
        model_lfn.intrinsics = intrinsics
        model_lfn.uv = uv
        model_lfn.num_iters = opt.num_inference_iters
        model_lfn.lr = opt.lr

        clean_acc_lfn = fb.accuracy(fmodel_lfn, rgb, labels)
        out_file_lfn.write(f"{c}: {clean_acc_lfn.item() * 100:4.1f}\n")

    # --- eval FF ---

    for imgs, labels in dataloader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        clean_acc_ff = fb.accuracy(fmodel_ff, imgs, labels)
        out_file_ff.write(f"{c}: {clean_acc_ff.item() * 100:4.1f}\n")


