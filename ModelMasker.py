from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import re
import copy
import pickle
from collections import namedtuple

'''
v1: Original version.
v2: Improved implementation to acount for batch normalization.
v3: Simplified implementation and added features.
    - Only the gradients of the masked model are computed by autograd.
        - The gradients of the model weights and mask weights are computed manually from those.
    - Adapt for use of any pytorch optimizer. Moved sgd behavior outside.
    - Added parameter warm-up inactive weights.
    - Divided mask_lr_decay into mask_l2_decay and mask_sl1_decay.
    - Removed abs_l1_reg_mask.
v3.1: Enable loading all the model but the last layer. Quick version before v4.



TODO:
    - Compare:
        - Start training masks at epoch 1 or 2.
        - Warm-up model weights or not.
        - Different resting points.
    - Optimize training for fixed mask.
    - New version to handle last layer separately. We don't want to mask it. But then what happens with logistic regression??
    - Add option to make mask random or not.
    - Data augmentation.
    

'''


def is_not_bn(x):
    return re.search(pattern=r'bn\d*\.(weight|bias)', string=x) is None

MaskerTrainingParameters = namedtuple('MaskerTrainingParameters', [
    'model_lr',
    'mask_lr',
    'model_l2_decay',
    'mask_resting_point',
    'mask_reg_factor',
    'warmup_inactive_weights'
])


class ModelMasker(nn.Module):
    """
    A class that adds masking functionalities to a pytorch model.
    
    The sgd updates are the following:
        model_weights -= model_lr * (model_weights.grad + model_decay * model_weights)
        l1_term = int(current_sparsity < sparsity)
        mask_weights -= mask_lr * (mask_weights.grad + mask_decay * (l1_term + mask_weights))
    """
    
    def __init__(self, model, device, masker_training_parameters, sparsity, train_mask,
                 unmaskable_parameters=[]):
        super(ModelMasker, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        self.mtp = masker_training_parameters
        self.sparsity = sparsity
        
        self.sigmoid = nn.Sigmoid()
        self.hardm = 1000
        self.softm = 10
        self.training_mask = train_mask
        self.unmaskable_parameters = unmaskable_parameters
        for pn in self.unmaskable_parameters:
            assert pn in dict(self.model.named_parameters())
        
        # maskable_parameters_names are all but bn and unmaskable_parameters.
        self.maskable_parameters_names = []
        for pn, pp in model.named_parameters():
            if is_not_bn(pn) and pn not in self.unmaskable_parameters:
                print('Masked:  ', pn, pp.numel())
                self.maskable_parameters_names.append(pn)
            else:
                print('Unmasked:', pn, pp.numel())
        
        # The weights of the model.
        self.model_weights = {}
        
        # The weights that generate the masks (before sigmoid).
        self.mask_weights = {}
        
        # Stats.
        self.steps = 0
        self.last_time_nonzero = {}
        self.densities = {'overall': []}
        self.reg_ratios = []
        
        # Initialize mask_weights.
        for pn, pp in self.get_maskable_parameters():
            self.model_weights[pn] = nn.Parameter(pp.data.clone())
            self.model_weights[pn].requires_grad = True
            #self.mask_weights[pn] = torch.randn(pp.shape).to(self.device) * 0.03 + 0.08 # + 0.1 instead
            self.mask_weights[pn] = nn.Parameter(torch.ones(pp.shape).to(self.device) * 0.1)
            self.mask_weights[pn].requires_grad = True
            self.last_time_nonzero[pn] = torch.zeros(pp.shape, dtype=int).to(self.device)
            self.densities[pn] = []
    
    # If is_model == True, returns all maskable model weights and bn weights.
    # If is_model == False, returns all mask weights.
    def get_optimizable_parameters(self, is_model):
        for pn, pp in self.model.named_parameters():
            if pn in self.maskable_parameters_names:
                yield self.model_weights[pn] if is_model else self.mask_weights[pn]
            elif is_model:
                yield pp

    def get_maskable_parameters(self):
        return filter(lambda x: x[0] in self.maskable_parameters_names, self.model.named_parameters())
    
    def state_dict(self):
        return {'model_state_dict': self.model.state_dict(),
                'model_weights': self.model_weights,
                'mask_weights': self.mask_weights}
    
    def load_state_dict(self, state_dict):
        assert set(state_dict['model_weights'].keys()) == set(self.maskable_parameters_names)
        self.model_weights = state_dict['model_weights']
        self.mask_weights = state_dict['mask_weights']
        
        for pn in self.unmaskable_parameters:
            del state_dict['model_state_dict'][pn]
        
        keys = self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
        assert set(keys.missing_keys) == set(self.unmaskable_parameters) and len(keys.unexpected_keys) == 0
    
    def load(self, path):
        with open(path, 'rb') as fp:
            self.load_state_dict(pickle.load(fp))

    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.state_dict(), fp)
    
    def count_ones(self, verbose=False):
        pn_stats = {}
        
        size_cnt = 0.
        ones_cnt = 0.
        for pn, pp in self.mask_weights.items():
            sc = pp.numel()
            oc = torch.sum(pp > 0).item()
            size_cnt += sc
            ones_cnt += oc
            if verbose:
                pn_stats[pn] = [oc, sc]
        
        if not verbose:
            return ones_cnt, ones_cnt*1./size_cnt
        else:
            pn_stats['overall'] = [ones_cnt, size_cnt]
            return pn_stats
    
    def shift_mask_weights(self, n_ones, randomize_mask=True):
        if randomize_mask:
            for pp in self.mask_weights.values():
                pp.data += torch.randn(pp.shape).to(self.device) * 0.03
        all_weights = [pp.data.flatten() for pp in self.mask_weights.values()]
        all_weights = torch.cat(all_weights, 0)
        values, _ = all_weights.topk(n_ones + 1, sorted=False)
        kth_value = values.min()
        for pn in self.mask_weights:
            self.mask_weights[pn].data -= kth_value
    
    def binarise(self, mask, hard):
        if not self.training or not self.training_mask:
            return 1. * (mask > 0)
        if hard:
            return self.sigmoid(mask * self.hardm)
        return self.sigmoid(mask * self.softm)
    
    def put_on_mask(self):
        for pn, pp in self.get_maskable_parameters():
            mask = self.binarise(self.mask_weights[pn].data, hard=True)
            pp.data = self.model_weights[pn].data * mask
    
    def forward(self, x):
        self.put_on_mask()
        return self.model(x)
    
    def compute_gradients(self):
        self.steps += 1
        assert(self.training)
        density_stats = self.count_ones(verbose=True)
        density = density_stats['overall'][0] * 1. / density_stats['overall'][1]
        self.densities['overall'].append(density)
        
        # We only move mask weights to -1 if the target sparsity hasn't been reached.
        should_decrease_mask = (1 - density) < self.sparsity
        
        mask_reg_grads = {}
        mask_grad_magnitude = 0
        mask_reg_magnitude = 0
        for pn, pp in self.get_maskable_parameters():
            if self.training_mask:
                # Set mask gradient without regularization term.
                soft_mask = self.binarise(self.mask_weights[pn].data, hard=False)
                mask_grad = pp.grad * self.model_weights[pn].data
                self.mask_weights[pn].grad = mask_grad * soft_mask * (1 - soft_mask)
                
                # Compute regularization statistics.
                if should_decrease_mask:
                    # reg term = (w - resting_point)^2 ==> grad reg term = 2 * (w - resting_point)
                    mask_reg_term = 2 * (self.mask_weights[pn].data - self.mtp.mask_resting_point)
                    mask_reg_grads[pn] = mask_reg_term
                    mask_grad_magnitude += self.mask_weights[pn].grad.abs().sum()
                    mask_reg_magnitude += mask_reg_grads[pn].abs().sum()
                    
                # Update stats.
                self.last_time_nonzero[pn][self.mask_weights[pn].data > 0] = self.steps
                self.densities[pn].append(density_stats[pn][0] * 1. / density_stats[pn][1])
            
            # Set model gradient with regularization.
            if self.training_mask and self.mtp.warmup_inactive_weights:
                model_weights_grad = pp.grad * soft_mask
            else:
                hard_mask = self.binarise(self.mask_weights[pn].data, hard=True)
                model_weights_grad = pp.grad * hard_mask
            model_reg_term = self.mtp.model_l2_decay * self.model_weights[pn]
            self.model_weights[pn].grad = model_weights_grad + model_reg_term
        
        # If needed, add regularization term to mask grads.
        if self.training_mask and should_decrease_mask:
            reg_ratio = mask_grad_magnitude / mask_reg_magnitude
            self.reg_ratios.append(reg_ratio)
            
            for pn, pp in self.get_maskable_parameters():
                self.mask_weights[pn].grad += self.mtp.mask_reg_factor * reg_ratio * mask_reg_grads[pn]

