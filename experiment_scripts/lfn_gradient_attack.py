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

p.add_argument('--attack_name', type=str, required=True)
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

#Classes that are being detected by LFN with accuracy >90%:
#selected_class_str = ["02691156", "04256520", "04530566", "03211117"]
if opt.single_class_string != None:
    selected_class_str = [opt.single_class_string]
else:
    selected_class_str = None

print("Loading dataset")
if selected_class_str is not None:
    class_id = [multiclass_dataio.class2string_dict[int(opt.single_class_string)]]
else:
    class_id = None
train_dataset = multiclass_dataio.SceneClassDataset(num_context=1, num_trgt=1,
                                                    root_dir=opt.data_root, query_sparsity=None,
                                                    img_sidelength=opt.img_sidelength, vary_context_number=True,
                                                    specific_observation_idcs=specific_observation_idcs, cache=None,
                                                    max_num_instances=opt.max_num_instances,
                                                    # max_num_observations_per_instance=opt.max_num_observations_train,
                                                    dataset_type=opt.set,
                                                    viewlist="/home/woody/iwi9/iwi9015h/light-field-networks/experiment_scripts/viewlists/src_dvr.txt",
                                                    num_instances_per_class=num_instances_per_class,
                                                    specific_classes=class_id)
dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False,
                          drop_last=True, num_workers=0)

print("Initializing model")
model = LFAutoDecoder(latent_dim=256, num_instances=train_dataset.num_instances, classify=True).cuda()
model.eval()

if opt.checkpoint_path is not None:
    print(f"Loading weights from {opt.checkpoint_path}...")
    state_dict = torch.load(opt.checkpoint_path)
    del state_dict['latent_codes.weight']
    model.load_state_dict(state_dict, strict=False)

fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

print("Running adversarial attacks")
assert len(dataloader) != 0

robust_accs = list()
for model_input, ground_truth in iter(dataloader): #will run infinitely
    #model_input, ground_truth = next(iter(dataloader)) # Dictionary
    inputs = model_input # (b, sidelength**2, 3)
    rgb = model_input['query']['rgb'].cuda()
    intrinsics = model_input['query']['intrinsics'].cuda()
    pose = model_input['query']['cam2world'].cuda()
    uv = model_input['query']['uv'].cuda().float()
    labels = model_input['query']['class'].squeeze().int().cuda() # (b)

    model.pose = pose
    model.intrinsics = intrinsics
    model.uv = uv
    model.num_iters = opt.num_inference_iters
    model.lr = opt.lr

    #attack = fb.attacks.GenAttack()
    #print('labels', labels)
    #print(f"clean accuracy:  {fb.accuracy(fmodel, rgb, labels) * 100:.1f} %") 
    attack = fb.attacks.L2FastGradientAttack()
    print(attack)

    epsilons = np.arange(20)/20*256

    out = attack(model=fmodel, inputs=rgb, criterion=fb.criteria.Misclassification(labels), epsilons=epsilons)
    print(out)

    # robust_accuracy = 1 - success.float().mean(axis=-1)
    # robust_accs.append(robust_accuracy)

    # if opt.out_file is not None:
    #     for (eps, acc) in zip(epsilons, robust_accuracy):
    #         out_file.write(f"{eps:<6} {acc.item() * 100:4.1f}\n")

    # else:
    #     print("robust accuracy for perturbations with")
    #     for (eps, acc) in zip(epsilons, robust_accuracy):
    #         print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
    #     print()
