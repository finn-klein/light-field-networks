import torch.nn as nn
from torch import Size


def image_loss(model_out, gt, mask=None):
    gt_rgb = gt['rgb']
    return nn.MSELoss()(gt_rgb, model_out['rgb']) * 200

def class_loss(model_out, gt, mask=None):
    gt_class = gt['class'].long().squeeze()
    # Add dimension if squeezed tensor is scalar (e.g. if batch_size is 1, which holds during validation)
    if gt_class.shape == Size([]):
        gt_class = gt_class.unsqueeze(0)
    pred_class = model_out['class']

    max_index = pred_class.max(dim=1)[1]
    correct = (max_index == gt_class).float()
    accuracy = correct.mean()

    return nn.CrossEntropyLoss(weight=None, size_average=None, reduce=None, reduction='mean')(pred_class, gt_class), accuracy

class LFLoss():
    def __init__(self, l2_weight=1, reg_weight=1e2):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_dict['img_loss'] = image_loss(model_out, gt)
        loss_dict['reg'] = (model_out['z']**2).mean() * self.reg_weight
        return loss_dict, {}

class LFClassLoss():
    def __init__(self, l2_weight=1., reg_weight=1e2, class_weight=1.):
        self.l2_weight = l2_weight
        self.reg_weight = reg_weight
        self.class_weight = class_weight

    def __call__(self, model_out, gt, model=None, val=False):
        loss_dict = {}
        loss_summaries = {}

        loss_dict['img_loss'] = image_loss(model_out, gt)  * self.l2_weight
        loss_dict['reg'] = (model_out['z']**2).mean() * self.reg_weight

        loss_dict['class_loss'], loss_summaries['accuracy'] = class_loss(model_out, gt)
        loss_dict['class_loss'] *= self.class_weight

        return loss_dict, loss_summaries

