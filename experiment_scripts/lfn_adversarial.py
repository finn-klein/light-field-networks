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
from models import LFAutoDecoder
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--data_root', required=True)
p.add_argument('--single_class_string', type=str, required=False)
p.add_argument('--checkpoint_path', type=str)

p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--img_sidelength', type=int, default=64, required=False)
p.add_argument('--batch_size', type=int, default=256)
p.add_argument('--num_inference_iters', type=int, default=150)
p.add_argument('--set', default='test')

p.add_argument('--specific_observation_idcs', type=str, default="22")
p.add_argument('--max_num_instances', type=int, default=256)
p.add_argument('--max_num_observations', type=int, default=256, required=False)
p.add_argument('--num_instances_per_class', type=int, required=False)
p.add_argument('--out_file', type=str, required=False)
p.add_argument('--eps', type=float, required=False)
p.add_argument('--adv_epsilon', type=float, default=1e-1)
p.add_argument('--out_folder', type=str, required=False)
opt = p.parse_args()

if opt.out_file is not None:
    out_file = open(opt.out_file, "w")

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

if opt.single_class_string != None:
    selected_class_str = [opt.single_class_string]
else:
    selected_class_str = None

print("Loading dataset")
if selected_class_str is not None:
    class_id = [multiclass_dataio.class2string_dict[int(opt.single_class_string)]]
else:
    class_id = None
dataset = multiclass_dataio.SceneClassDataset(num_context=1, num_trgt=1,
                                                    root_dir=opt.data_root, query_sparsity=None,
                                                    img_sidelength=opt.img_sidelength, vary_context_number=True,
                                                    specific_observation_idcs=specific_observation_idcs, cache=None,
                                                    max_num_instances=opt.max_num_instances,
                                                    # max_num_observations_per_instance=opt.max_num_observations_train,
                                                    dataset_type=opt.set,
                                                    viewlist="/home/woody/iwi9/iwi9015h/light-field-networks/experiment_scripts/viewlists/src_dvr.txt",
                                                    num_instances_per_class=num_instances_per_class,
                                                    specific_classes=class_id)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                          drop_last=True, num_workers=0)

print("Initializing model")
model = LFAutoDecoder(latent_dim=256, num_instances=dataset.num_instances, classify=True).cuda()
model.eval()

if opt.checkpoint_path is not None:
    print(f"Loading weights from {opt.checkpoint_path}...")
    state_dict = torch.load(opt.checkpoint_path)
    del state_dict['latent_codes.weight']
    model.load_state_dict(state_dict, strict=False)

for (model_input, ground_truth) in iter(dataloader):
    # def adversarial_attack(self, rgb, pose, intrinsics, uv, max_epsilon=1e-1, num_adv_iters=100, adv_lr=1e-3):
    inputs = model_input
    rgb = model_input['query']['rgb'].cuda()
    intrinsics = model_input['query']['intrinsics'].cuda()
    pose = model_input['query']['cam2world'].cuda()
    uv = model_input['query']['uv'].cuda().float()
    labels = model_input['query']['class'].squeeze().cuda()
    model.lr = opt.lr
    model.num_iters = num_iters

    epsilon = opt.adv_epsilon
    model.adversarial_attack(rgb, labels, pose, intrinsics, uv, epsilon, opt.out_folder)