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

p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--img_sidelength', type=int, default=64, required=False)
p.add_argument('--batch_size', type=int, default=20)
p.add_argument('--num_inference_iters', type=int, default=150)
p.add_argument('--data_root', required=True)
               # default='/om2/user/egger/MultiClassSRN/data/NMR_SmallDatasetForIOTesting')
p.add_argument('--set', default='test')
p.add_argument('--checkpoint_path', type=str,
               default='/om2/user/sitzmann/logs/light_fields/adv_test/64_128_None/checkpoints/model_current.pth')
p.add_argument('--specific_observation_idcs', type=str, default=None)
p.add_argument('--max_num_instances', type=int, default=-1)
p.add_argument('--max_num_observations', type=int, default=50, required=False)
opt = p.parse_args()

lr = opt.lr
num_iters = opt.num_inference_iters

if opt.specific_observation_idcs is not None:
    specific_observation_idcs = util.parse_comma_separated_integers(opt.specific_observation_idcs)
else:
    specific_observation_idcs = None

#Classes that are being detected by LFN with accuracy >90%:
selected_class_str = ["02691156", "04256520", "04530566", "03211117"]

print("Loading dataset")
train_dataset = multiclass_dataio.SceneClassDataset(num_context=0, num_trgt=1,
                                                    root_dir=opt.data_root, query_sparsity=None,
                                                    img_sidelength=opt.img_sidelength, vary_context_number=True,
                                                    specific_observation_idcs=specific_observation_idcs, cache=None,
                                                    max_num_instances=opt.max_num_instances,
                                                    # max_num_observations_per_instance=opt.max_num_observations_train,
                                                    dataset_type=opt.set,
                                                    specific_classes=selected_class_str)
dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                          drop_last=True, num_workers=0)

print("Initializing model")
model = LFAutoDecoder(latent_dim=256, num_instances=train_dataset.num_instances, classify=True, lr=lr).cuda()
model.eval()

if opt.checkpoint_path is not None:
    print(f"Loading weights from {opt.checkpoint_path}...")
    state_dict = torch.load(opt.checkpoint_path)
    del state_dict['latent_codes.weight']
    model.load_state_dict(state_dict, strict=False)

fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
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

#attack = fb.attacks.GenAttack()
print('labels', labels)
print(f"clean accuracy:  {fb.accuracy(fmodel, rgb, labels) * 100:.1f} %")
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
# epsilons = [1.0]
print('labels', labels.size())
raw_advs, clipped_advs, success = attack(fmodel, inputs=rgb, criterion=labels, epsilons=epsilons)
robust_accuracy = 1 - success.float().mean(axis=-1)
print("robust accuracy for perturbations with")
for eps, acc in zip(epsilons, robust_accuracy):
    print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
print()
print("we can also manually check this:")
print()
print("robust accuracy for perturbations with")
for eps, advs_ in zip(epsilons, clipped_advs):
    acc2 = fb.accuracy(fmodel, advs_, labels)
    print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
    print("    perturbation sizes:")
    perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
    print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
    if acc2 == 0:
        break