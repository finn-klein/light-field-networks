import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np

import util

import conv_modules
import custom_layers
import geometry
import hyperlayers
from loss_functions import LFClassLoss

from torch.optim.lr_scheduler import LambdaLR

from PIL import Image


class LightFieldModel(nn.Module):
    def __init__(self, latent_dim, parameterization='plucker', network='relu',
                 fit_single=False, conditioning='hyper', depth=False, alpha=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 256
        self.fit_single = fit_single
        self.parameterization = parameterization
        self.conditioning = conditioning
        self.depth = depth
        self.alpha = alpha

        out_channels = 3

        if self.depth:
            out_channels += 1
        if self.alpha:
            out_channels += 1
            self.background = torch.ones((1, 1, 1, 3)).cuda()

        if self.fit_single or conditioning in ['hyper', 'low_rank']:
            if network == 'relu':
                self.phi = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=6,
                                                 in_features=6, out_features=out_channels, outermost_linear=True, norm='layernorm_na')
            elif network == 'siren':
                omega_0 = 30.
                self.phi = custom_layers.Siren(in_features=6, hidden_features=256, hidden_layers=8,
                                               out_features=out_channels, outermost_linear=True, hidden_omega_0=omega_0,
                                               first_omega_0=omega_0)
        elif conditioning == 'concat':
            self.phi = nn.Sequential(
                nn.Linear(6+self.latent_dim, self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                nn.Linear(self.num_hidden_units_phi, 3)
            )

        if not self.fit_single:
            if conditioning=='hyper':
                self.hyper_phi = hyperlayers.HyperNetwork(hyper_in_features=self.latent_dim,
                                                          hyper_hidden_layers=1,
                                                          hyper_hidden_features=self.latent_dim,
                                                          hypo_module=self.phi)
            elif conditioning=='low_rank':
                self.hyper_phi = hyperlayers.LowRankHyperNetwork(hyper_in_features=self.latent_dim,
                                                                 hyper_hidden_layers=1,
                                                                 hyper_hidden_features=512,
                                                                 hypo_module=self.phi,
                                                                 nonlinearity='leaky_relu')

        print(self.phi)
        print(np.sum(np.prod(param.shape) for param in self.phi.parameters()))

    def get_light_field_function(self, z=None):
        if self.fit_single:
            phi = self.phi
        elif self.conditioning in ['hyper', 'low_rank']:
            phi_weights = self.hyper_phi(z)
            phi = lambda x: self.phi(x, params=phi_weights)
        elif self.conditioning == 'concat':
            def phi(x):
                b, n_pix = x.shape[:2]
                z_rep = z.view(b, 1, self.latent_dim).repeat(1, n_pix, 1)
                return self.phi(torch.cat((z_rep, x), dim=-1))
        return phi

    def get_query_cam(self, input):
        query_dict = input['query']
        pose = util.flatten_first_two(query_dict["cam2world"])
        intrinsics = util.flatten_first_two(query_dict["intrinsics"])
        uv = util.flatten_first_two(query_dict["uv"].float())
        return pose, intrinsics, uv

    def forward(self, input, val=False, compute_depth=False, timing=False):
        out_dict = {}
        query = input['query']
        b, n_ctxt = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]

        if not self.fit_single:
            if 'z' in input:
                z = input['z']
            else:
                z = self.get_z(input)

            out_dict['z'] = z
            z = z.view(b * n_qry, self.latent_dim)

        query_pose, query_intrinsics, query_uv = self.get_query_cam(input)

        if self.parameterization == 'plucker':
            light_field_coords = geometry.plucker_embedding(query_pose, query_uv, query_intrinsics)
        else:
            ray_origin = query_pose[:, :3, 3][:, None, :]
            ray_dir = geometry.get_ray_directions(query_uv, query_pose, query_intrinsics)
            intsec_1, intsec_2 = geometry.ray_sphere_intersect(ray_origin, ray_dir, radius=100)
            intsec_1 = F.normalize(intsec_1, dim=-1)
            intsec_2 = F.normalize(intsec_2, dim=-1)

            light_field_coords = torch.cat((intsec_1, intsec_2), dim=-1)
            out_dict['intsec_1'] = intsec_1
            out_dict['intsec_2'] = intsec_2
            out_dict['ray_dir'] = ray_dir
            out_dict['ray_origin'] = ray_origin

        light_field_coords.requires_grad_(True)
        out_dict['coords'] = light_field_coords.view(b*n_qry, n_pix, 6)

        lf_function = self.get_light_field_function(None if self.fit_single else z)
        out_dict['lf_function'] = lf_function

        if timing: t0 = time.time()
        lf_out = lf_function(out_dict['coords'])
        if timing: t1 = time.time(); total_n = t1 - t0; print(f'{total_n}')

        rgb = lf_out[..., :3]

        if self.depth:
            depth = lf_out[..., 3:4]
            out_dict['depth'] = depth.view(b, n_qry, n_pix, 1)

        rgb = rgb.view(b, n_qry, n_pix, 3)

        if self.alpha:
            alpha = lf_out[..., -1:].view(b, n_qry, n_pix, 1)
            weight = 1 - torch.exp(-torch.abs(alpha))
            rgb = weight * rgb + (1 - weight) * self.background
            out_dict['alpha'] = weight

        if compute_depth:
            with torch.enable_grad():
                lf_function = self.get_light_field_function(z)
                depth = util.light_field_depth_map(light_field_coords, query_pose, lf_function)['depth']
                depth = depth.view(b, n_qry, n_pix, 1)
                out_dict['depth'] = depth

        out_dict['rgb'] = rgb

        pred_class = self.linear_classifier(z)
        out_dict['class'] = pred_class
        return out_dict


class LFAutoDecoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', classify=False, out_path=None, **kwargs):
        super().__init__(latent_dim=latent_dim, parameterization=parameterization, **kwargs)
        self.num_instances = num_instances

        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        self.linear_classifier = nn.Linear(self.latent_dim, 13)
        self.linear_classifier.apply(custom_layers.init_weights_normal)

        self.pose = None
        self.intrinsics = None
        self.uv = None

        if classify:
            self.forward = self.forward_classify
        self.out_path = out_path

    def get_z(self, input, val=False):
        instance_idcs = input['query']["instance_idx"].long()
        z = self.latent_codes(instance_idcs)
        return z

    def forward_render(self, latents, pose, uv, intrinsics, b, n_qry, n_pix):
        """
        Helper function to render image from latent code and camera information
        """
        light_field_coords = geometry.plucker_embedding(pose, uv, intrinsics)
        phi = self.get_light_field_function(latents)
        lf_out = phi(light_field_coords)
        novel_views = lf_out[..., :3]
        return novel_views.view(b, n_qry, n_pix, 3)

    def infer_and_classify(self, rgb, pose, intrinsics, uv, labels=None, detailed_logging=False, return_latents=False):
        b, n_ctxt = uv.shape[:2]
        n_qry, n_pix = uv.shape[1:3]

        pose = util.flatten_first_two(pose)
        intrinsics = util.flatten_first_two(intrinsics)
        uv = util.flatten_first_two(uv)

        if self.out_path is not None:
            f = open(self.out_path, "a")

        with torch.enable_grad():
            num_instances = rgb.shape[0]
            latent_codes = nn.Embedding(num_instances, self.latent_dim).cuda() # num_instances, self.latent_dim
            nn.init.zeros_(latent_codes.weight)

            optimizer = torch.optim.Adam(params = [latent_codes.weight], lr = self.lr)
            #lr_schedule = lambda epoch: 0.1**(epoch/self.num_iters)
            #scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

            for iter in range(self.num_iters):
                novel_views = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)
                # light_field_coords = geometry.plucker_embedding(pose, uv, intrinsics)
                # phi = self.get_light_field_function(latent_codes.weight)
                # lf_out = phi(light_field_coords)
                # novel_views = lf_out[..., :3]
                # novel_views = novel_views.view(b, n_qry, n_pix, 3)

                loss = nn.MSELoss()(novel_views, rgb) * 200 + torch.mean(latent_codes.weight**2) * 100 # reg_weight = 100
                #print("Epoch", iter)
                if detailed_logging:
                    with torch.no_grad():
                        print("Latent:", latent_codes.weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()

                if self.out_path is not None and labels is not None and iter % 50 == 0:
                    with torch.no_grad():
                        pred_class = self.linear_classifier(latent_codes.weight).argmax(axis=-1)
                        acc = float((pred_class == labels).float().mean(axis=-1).cpu())
                        f.write(f"iter {iter}: {acc}\n")

                if iter % 100 == 0:
                    print(f"Inference iter {iter}, loss {loss}.")
        pred_class = self.linear_classifier(latent_codes.weight)
        print("predictions", pred_class.argmax(axis=-1))
        if self.out_path is not None:
            f.close()
        if return_latents:
            return pred_class, latent_codes.weight
        return pred_class

    def forward_classify(self, images):
        return self.infer_and_classify(images, self.pose, self.intrinsics, self.uv)

    def forward_w_dict(self, input, lr=1e-3, num_iters=150):
        rgb = input["rgb"]
        pose = input["pose"]
        intrinsics = input["intrinsics"]
        uv = input["uv"].float()
        return self.infer_and_classify(rgb, pose, intrinsics, uv)

    def adversarial_attack(self, rgb, labels, pose, intrinsics, uv, epsilons=1e-1, num_adv_iters=50, adv_lr=2e-4, out_folder=None, save_imgs = False, out_file=None):
        # labels === ground truth labels
        b, n_ctxt = uv.shape[:2]
        n_qry, n_pix = uv.shape[1:3]

        pose = util.flatten_first_two(pose)
        intrinsics = util.flatten_first_two(intrinsics)
        uv = util.flatten_first_two(uv)

        if self.out_path is not None:
            f = open(self.out_path, "a")
        
        if out_folder is not None and out_file is not None:
            out_file = open(f"{out_folder}/{out_file}.txt", "w")

        # latent code optimization
        with torch.enable_grad():
            num_instances = rgb.shape[0]
            latent_codes = nn.Embedding(num_instances, self.latent_dim).cuda() # num_instances, self.latent_dim
            nn.init.zeros_(latent_codes.weight)

            optimizer = torch.optim.Adam(params = [latent_codes.weight], lr = self.lr)

            for iter in range(self.num_iters):
                novel_views = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)

                loss = nn.MSELoss()(novel_views, rgb) * 200 + torch.mean(latent_codes.weight**2) * 100 # reg_weight = 100
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pred_class = self.linear_classifier(latent_codes.weight)
        pred_class = pred_class.argmax(axis=-1)
        clean_acc = float((pred_class == labels).float().mean(axis=-1).cpu())
        print(f"Clean accuracy: {clean_acc * 100:.1f}%")

        # Save clean latents to restore later
        clean_latents = latent_codes.weight.data.clone()
        clean_render = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)

        print("Running adversarial attacks")
        
        for eps in epsilons:
            print(f"epsilon: {eps}")
            # Class weight is -1 because for adversarial attacks we want to increase class loss during gradient descent
            loss_function = LFClassLoss(l2_weight=1., reg_weight=1e2, class_weight=-1.)
            optimizer = torch.optim.Adam(params = [latent_codes.weight], lr = adv_lr)
            mask = [False]*self.num_instances
            for iter in range(num_adv_iters):
                novel_views = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)
                pred_class = self.linear_classifier(latent_codes.weight)
                pred = {"rgb": novel_views, "class": pred_class, "z": latent_codes.weight}
                gt = {"rgb": rgb, "class": labels}

                losses, _ = loss_function(pred, gt)
                total_loss = 0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    total_loss += single_loss
                optimizer.zero_grad()
                total_loss.backward()

                #backup old latents (guaranteed to be within bounds) before optimization step
                old_latents = latent_codes.weight.data.clone() # Want these to not be affected by optimizer
                old_latents.requires_grad_(False) # not sure if this is necessary

                def batched_l2_distance(x, y, axes):
                    """Compute the batched l2 distance of x to y along the described axes"""
                    axes.sort()
                    axes = axes[::-1]
                    norm = 1
                    result = (x - y).pow(2)
                    for axis in axes:
                        norm *= result.shape[axis]
                        result = result.sum(axis)
                    #return 1/norm * result.sqrt().flatten()
                    return result.sqrt().flatten()

                # During first iteration, try to adjust LR such that not all latents are immediately moved outside the limit
                terminate = False
                cnt = 0
                while (not terminate and iter == 0 and cnt < 50):
                    cnt += 1
                    optimizer.step()
                    novel_views = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)

                    distance = batched_l2_distance(novel_views, clean_render, [2, 3])
                    mask = (distance > eps)
                    print(distance)
                    print(mask)
                    terminate = not mask.any()
                    # half optimizer LR, restore and retry
                    optimizer.param_groups[0]['lr'] /= 2
                    latent_codes.weight.data[mask, :] = old_latents[mask, :]
                    print(f"lr: {optimizer.param_groups[0]['lr']}")

                # if not mask.any():
                #     break

                # During any other iteration, assume we have correctly adjusted the LR
                optimizer.step()

                novel_views = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)
                distance = batched_l2_distance(novel_views, rgb, [2, 3])
                mask = (distance > eps)
                #restore those latents which have been moved out of bounds
                latent_codes.weight.data[mask, :] = old_latents[mask, :]
                # If all latents have been moved out of bounds, we can abort the optimization
                if mask.all():
                    break

                print(f"----- Iteration {iter} -----")
                print(f"{mask.sum().item()}/{num_instances} latents have reached the limit")
                #print("")
            
            # if not mask.any():
            #     out_file.write(f"{eps}: all latents out of bounds")
            #     # continue with next epsilon
            #     continue
            
            #n_latents_in_bound = (not mask).sum()
            adv_pred_class = self.linear_classifier(latent_codes.weight)
            adv_pred_class = adv_pred_class.argmax(axis=-1)
            # only remember those correct predictions which had an underlying latent within bounds
            correct_predictions = (adv_pred_class == labels)

            # save all misclassifications
            if out_folder is not None and save_imgs:
                # final render
                adv_rgb = self.forward_render(latent_codes.weight, pose, uv, intrinsics, b, n_qry, n_pix)
                for i in range(self.num_instances):
                    if not correct_predictions[i]:
                        adv_img = adv_rgb[i, :, :, :].squeeze(1).reshape([64, 64, 3]).cpu().detach().numpy()*255
                        adv_img = adv_img.astype(np.uint8)
                        gt_img = rgb[i, :, :, :].squeeze(1).reshape([64, 64, 3]).cpu().detach().numpy()*255
                        gt_img = gt_img.astype(np.uint8)
                        Image.fromarray(adv_img).save(f"{out_folder}/adv_{i}.png", mode="RGB")
                        Image.fromarray(gt_img).save(f"{out_folder}/gt_{i}.png", mode="RGB")

            # TODO: Replace with sum (along which axis? idk rn too tired)
            adv_acc = float(correct_predictions.float().mean(axis=-1).cpu())
            if out_folder is not None:
                out_file.write(f"{eps}: {adv_acc}\n")
            print(f"Adversarial accuracy: {adv_acc * 100:.1f}%")

            # restore clean latents for next epsilon
            latent_codes.weight.data = clean_latents



class LFEncoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', conditioning='hyper'):
        super().__init__(latent_dim, parameterization, conditioning='low_rank')
        self.num_instances = num_instances
        self.encoder = conv_modules.Resnet18(c_dim=latent_dim)

    def get_z(self, input, val=False):
        n_qry = input['query']['uv'].shape[1]
        rgb = util.lin2img(util.flatten_first_two(input['context']['rgb']))
        z = self.encoder(rgb)
        z = z.unsqueeze(1).repeat(1, n_qry, 1)
        z *= 1e-2
        return z
