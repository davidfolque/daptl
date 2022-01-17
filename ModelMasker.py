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

Next version:
    v3: Simplified implementation. Now only the gradients of the masked model are computed by autograd.
    - Remove abs_l1_reg_mask. Done.
    - Add 2 regularization parameters. Done.
    - Add epsilon to mask gradients
    - Add option to make mask random or not.

TODO:
    - Adapt for the use of any optimizer:
        - Save weights as parameters, create function to compute their grads manually.

'''



'''
Model updates: w -= model_lr * (w.grad + model_l2_decay * w)
Mask updates: w -= mask_lr * (dL/d(mask) * (mask_grad_eps + d(masked_model)/d())
'''

MaskerTrainingParameters = namedtuple('MaskerTrainingParameters', [
    'model_lr',
    'mask_lr',
    'model_l2_decay',
    'mask_l2_decay',
    'mask_sl1_decay',
    'mask_grad_eps'
])


class ModelMasker(nn.Module):
    """
    A class that adds masking functionalities to a pytorch model.
    
    The sgd updates are the following:
        model_weights -= model_lr * (model_weights.grad + model_decay * model_weights)
        l1_term = int(current_sparsity < sparsity)
        mask_weights -= mask_lr * (mask_weights.grad + mask_decay * (l1_term + mask_weights))
    """
    
    def __init__(self, model, device, masker_training_parameters, sparsity):
        super(ModelMasker, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        self.mtp = masker_training_parameters
        self.sparsity = sparsity
        
        self.sigmoid = nn.Sigmoid()
        self.hardm = 1000
        self.softm = 10
        self.training_mask = True
        
        # The weights of the model.
        self.model_weights = {}
        
        # The weights that generate the masks (before sigmoid).
        self.mask_weights = {}
        
        # The masks generated by the hard sigmoid. Kept for their gradient.
        # Is not none between a forward and a sgd_step.
        self.masks = None
        
        # The last step in which each weight was not zero.
        self.steps = 0
        self.last_time_nonzero = {}
        
        # Initialize mask_weights.
        for pn, pp in self.get_maskable_parameters():
            self.model_weights[pn] = pp.data.clone()
            self.model_weights[pn].requires_grad = False
            #self.mask_weights[pn] = torch.randn(pp.shape).to(self.device) * 0.03 + 0.08 # + 0.1 instead
            self.mask_weights[pn] = torch.ones(pp.shape).to(self.device) * 0.01
            self.last_time_nonzero[pn] = torch.zeros(pp.shape, dtype=int).to(self.device)
    
    def multiply_learning_rates(self, factor):
        self.mtp = self.mtp._replace(model_lr=self.mtp.model_lr * factor,
                                     mask_lr=self.mtp.mask_lr * factor)

    def get_maskable_parameters(self):
        def is_not_bn(x):
            return re.search(pattern=r'bn\d*\.(weight|bias)', string=x[0]) is None
        return filter(is_not_bn, self.model.named_parameters())
    
    def state_dict(self):
        return {'model_state_dict': self.model.state_dict(),
                'model_weights': self.model_weights,
                'mask_weights': self.mask_weights}
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model_weights = state_dict['model_weights']
        self.mask_weights = state_dict['mask_weights']
    
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
            pn_stats['overall'] = ones_cnt*1./size_cnt
            return pn_stats
        
    # Should be called between forward and sgd_step.
    def count_ones_differentiable(self):
        size_cnt = 0
        ones_cnt = 0
        for pn, pp in self.mask_weights.items():
            size_cnt += pp.numel()
            ones_cnt += self.masks[pn].sum()
        return size_cnt, ones_cnt
    
    def shift_mask_weights(self, n_ones):
        all_weights = [pp.flatten() for pp in self.mask_weights.values()]
        all_weights = torch.cat(all_weights, 0)
        values, _ = all_weights.topk(n_ones + 1, sorted=False)
        kth_value = values.min()
        for pn in self.mask_weights:
            self.mask_weights[pn] -= kth_value
    
    def binarise(self, mask, hard):
        if not self.training or not self.training_mask:
            return 1. * (mask > 0)
        if hard:
            return self.sigmoid(mask * self.hardm)
        return self.sigmoid(mask * self.softm)
    
    def put_on_mask(self):
        assert(self.masks is None)
        self.masks = {}
        for pn, pp in self.get_maskable_parameters():
            self.masks[pn] = self.binarise(self.mask_weights[pn], hard=True)
            pp.data = self.model_weights[pn] * self.masks[pn]
    
    def forward(self, x):
        self.put_on_mask()
        result = self.model(x)
        if not self.training:
            self.masks = None
        return result
    
    def sgd_step(self):
        self.steps += 1
        assert(self.training)
        _, density = self.count_ones()
        mask_grad_norm = 0
        mask_decay_norm = 0
        total = 0
        for pn, pp in self.get_maskable_parameters():
            if self.training_mask:
                soft_mask = self.binarise(self.mask_weights[pn], hard=False)
                masks_grad = pp.grad * self.model_weights[pn]
                gradient_term = masks_grad * soft_mask * (1 - soft_mask)
                
                # We only move mask weights to -1 if the target sparsity hasn't been reached.
                decay_term = self.mtp.mask_l2_decay * self.mask_weights[pn]
                if (1 - density) < self.sparsity:
                    decay_term += self.mtp.mask_sl1_decay
                
                
                self.mask_weights[pn] -= self.mtp.mask_lr * (gradient_term + decay_term)
                mask_grad_norm += gradient_term.abs().sum().item()
                mask_decay_norm += decay_term.abs().sum().item()
                total += gradient_term.numel()
                
                # Update last_time_nonzero.
                self.last_time_nonzero[pn][self.mask_weights[pn] > 0] = self.steps
            
            #model_weights_grad = pp.grad * self.masks[pn]
            model_weights_grad = pp.grad * soft_mask
            gradient_term = model_weights_grad
            decay_term = self.mtp.model_l2_decay * self.model_weights[pn]
            self.model_weights[pn] -= self.mtp.model_lr * (gradient_term + decay_term)
            
        self.masks = None
        #print('grad ', mask_grad_norm / total)
        #print('decay', mask_decay_norm / total)

