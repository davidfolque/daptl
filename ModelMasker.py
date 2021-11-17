from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import copy
import pickle

class ModelMasker(nn.Module):
    def __init__(self, original_model, device, model_lr, mask_lr, decay):
        super(ModelMasker, self).__init__()
        
        self.ones_model = original_model
        self.device = device
        self.model_lr = model_lr
        self.mask_lr = mask_lr
        self.decay = decay
        
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
        
        # The model with the masks put on. Is not none between a forward and a sgd_step.
        self.masked_model = None
        
        # Initialize mask_weights.
        for pn, pp in self.ones_model.named_parameters():
            pp.requires_grad = False
            self.model_weights[pn] = pp.clone()
            self.model_weights[pn].requires_grad = True
            self.mask_weights[pn] = torch.randn(pp.shape).to(self.device) * 0.03 + 0.05 # + 0.1 instead
            pp.data = torch.ones(pp.shape)
    
    def zero_grad(self):
        super(ModelMasker, self).zero_grad()
        for pn in self.model_weights:
            if self.model_weights[pn].grad is not None:
                self.model_weights[pn].grad *= 0
            if self.mask_weights[pn].grad is not None:
                self.mask_weights[pn].grad *= 0
    
    def state_dict(self):
        return {'model_weights': self.model_weights, 'mask_weights': self.mask_weights}
    
    def load_state_dict(self, state_dict):
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
        assert(self.masked_model is None)
        self.masks = {}
        self.masked_model = copy.deepcopy(self.ones_model)
        for pn, pp in self.masked_model.named_parameters():
            self.masks[pn] = self.binarise(self.mask_weights[pn], hard=True)
            self.masks[pn].requires_grad = self.training_mask
            self.model_weights[pn].requires_grad = True
            pp *= self.model_weights[pn] * self.masks[pn]
    
    def forward(self, x):
        self.put_on_mask()
        result = self.masked_model(x)
        if not self.training:
            self.masks = None
            self.masked_model = None
        return result
    
    def sgd_step(self):
        assert(self.training)
        for pn, pp in self.masked_model.named_parameters():
            if self.training_mask:
                soft_mask = self.binarise(self.mask_weights[pn], hard=False).detach()
                gradient_term = self.masks[pn].grad * soft_mask * (1 - soft_mask)
                
                decay_term = self.decay * (1 + self.mask_weights[pn])
                self.mask_weights[pn] -= self.mask_lr * (gradient_term + decay_term)
            
            gradient_term = self.model_weights[pn].grad
            self.model_weights[pn].requires_grad = False
            decay_term = self.decay * self.model_weights[pn]
            self.model_weights[pn] -= self.model_lr * (gradient_term + decay_term)
            
        self.masks = None
        self.masked_model = None

