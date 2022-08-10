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

import multiclass_dataio
from torch.utils.data import DataLoader
from models import LFAutoDecoder
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', required=False, default="/home/vault/iwi9/iwi9015h/light_fields/acc_per_iter")
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--img_sidelength', type=int, default=64, required=False)
p.add_argument('--batch_size', type=int, default=256)
p.add_argument('--num_inference_iters', type=int, default=150)
p.add_argument('--data_root', required=True)
               # default='/om2/user/egger/MultiClassSRN/data/NMR_SmallDatasetForIOTesting')
p.add_argument('--set', default='test')
p.add_argument('--checkpoint_path', type=str,
               default='/om2/user/sitzmann/logs/light_fields/adv_test/64_128_None/checkpoints/model_current.pth')
p.add_argument('--specific_observation_idcs', type=str, default=None)
p.add_argument('--max_num_instances', type=int, default=-1)
p.add_argument('--max_num_observations', type=int, default=50, required=False)
p.add_argument('--num_instances_per_class', type=int, required=True)
p.add_argument('--single_class_index', type=str, required=False)
opt = p.parse_args()

lr = opt.lr
num_iters = opt.num_inference_iters

if opt.specific_observation_idcs is not None:
    specific_observation_idcs = util.parse_comma_separated_integers(opt.specific_observation_idcs)
else:
    specific_observation_idcs = None

if opt.num_instances_per_class is not None:
    num_instances_per_class = opt.num_instances_per_class
    max_num_instances = num_instances_per_class
else:
    num_instances_per_class = None
    max_num_instances = opt.max_num_instances

class_id = multiclass_dataio.class2string_dict[int(opt.single_class_index)]
out_path = opt.logging_root + "/" + class_id + ".txt"

print("Initializing model")
model = LFAutoDecoder(latent_dim=256, num_instances=opt.num_instances_per_class, classify=True, out_path=out_path).cuda()
model.eval()

if opt.checkpoint_path is not None:
    print(f"Loading weights from {opt.checkpoint_path}...")
    state_dict = torch.load(opt.checkpoint_path)
    del state_dict['latent_codes.weight']
    model.load_state_dict(state_dict, strict=False)

print(f"Loading dataset for class {class_id} (index {multiclass_dataio.string2class_dict[class_id]})")
train_dataset = multiclass_dataio.SceneClassDataset(num_context=0, num_trgt=1,
                                                    root_dir=opt.data_root, query_sparsity=None,
                                                    img_sidelength=opt.img_sidelength, vary_context_number=True,
                                                    specific_observation_idcs=specific_observation_idcs, cache=None,
                                                    max_num_instances=max_num_instances,
                                                    # max_num_observations_per_instance=opt.max_num_observations_train,
                                                    dataset_type=opt.set,
                                                    specific_classes=[class_id],
                                                    num_instances_per_class=num_instances_per_class)
dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

model_input, ground_truth = next(iter(dataloader)) # Dictionary
inputs = model_input # (b, sidelength**2, 3)
rgb = model_input['query']['rgb'].cuda()
intrinsics = model_input['query']['intrinsics'].cuda()
pose = model_input['query']['cam2world'].cuda()
uv = model_input['query']['uv'].cuda().float()
labels = model_input['query']['class'].squeeze().cuda() # (b)

model.pose = pose
model.intrinsics = intrinsics
model.uv = uv
model.num_iters = opt.num_inference_iters
model.lr = opt.lr

# inference with logging
model.infer_and_classify(rgb, pose, intrinsics, uv, labels=labels)