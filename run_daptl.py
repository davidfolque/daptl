from __future__ import print_function
import argparse
import logging
import copy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18

from ModelMasker import ModelMasker, MaskerTrainingParameters
from Nets import LogReg, Conv1, Conv2, Conv3, WrapperNet
from Tasks import get_datasets

from GridRun import grid_run
from Persistence import Persistence


def train(args, model, model_optim, mask_optim, model_lr_scheduler, mask_lr_scheduler, device, 
          train_loader, epoch, verbose=True, progress_bar=False):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, disable=not progress_bar)):
        #if batch_idx > 50:
            #break
        data, target = data.to(device), target.to(device)
        model.zero_grad() # optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target) # Accepts log-probabilities.
        train_loss += loss.item() * len(data)
        loss.backward()
        model.compute_gradients()
        model_optim.step()
        mask_optim.step()
        if model_lr_scheduler:
            model_lr_scheduler.step()
        if mask_lr_scheduler:
            mask_lr_scheduler.step()
        ones, density = model.count_ones()
        if verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDensity: {:.6f} ({})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), density, int(ones)))
    
    # Loss, accuracy
    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def test(model, device, test_loader, verbose=True, progress_bar=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, disable=not progress_bar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    # Loss, accuracy
    return test_loss, correct / len(test_loader.dataset)
    


def main(args=None, return_model=False):
    logging.basicConfig(level=logging.INFO)
    
    # Training settings
    parser = argparse.ArgumentParser(description='Differentiable Architecture Prunning for Transfer Learning')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model-lr', type=float, default=0.05, metavar='Model LR',
                        help='model learning rate (default: 0.05)')
    parser.add_argument('--mask-lr', type=float, default=0.05, metavar='Mask LR',
                        help='mask learning rate (default: 0.05)')
    parser.add_argument('--model-decay', type=float, default=1e-4, metavar='Model weight decay',
                        help='L2 and model weight decay (default: 1e-4)')
    parser.add_argument('--mask-resting-point', type=float, default=-1, metavar='Mask resting point parameter')
    parser.add_argument('--mask-reg-factor', type=float, default=-1, metavar='Mask regularization factor')
    parser.add_argument('--sparsity', type=str, default='[0.5, 0.7, 0.8, 0.9, 0.95]', metavar='Sparsity',
                        help='Proportion of the weights to zero out (default: [0.5, 0.7, 0.8, 0.9, 0.95])')
    parser.add_argument('--lr-factor', type=float, default=0.9, metavar='M',
                        help='Learning rate scheduler factor (default: 0.9)')
    parser.add_argument('--seed', type=str, default='[1,2,3,4,5]', metavar='S',
                        help='random seeds (default: 1-5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--task-type', choices=['upstream', 'downstream'], required=True)
    parser.add_argument('--persistence', type=str)
    parser.add_argument('--no-persistence', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--progress-bar', action='store_true', default=False)
    parser.add_argument('--plots', type=str, help='\'show\' to show in notebook/terminal, or a path to store images')
    parser.add_argument('--few-shot', action='store_true', default=False)
    parser.add_argument('--few-shot-size', type=int, default=10)
    parser.add_argument('--task', nargs='+', choices=['mnist1','mnist2','mnist3','cifar-32', 'cifar-224'], required=True)
    parser.add_argument('--model', choices=['LogReg', 'Conv1', 'Conv2', 'Conv3', 'ResNet'], default='LogReg')
    parser.add_argument('--late-mask-training', action='store_true', default=False)
    args = parser.parse_args(args=args)
    
    if args.persistence is None and not args.no_persistence:
        print('Either specify --persistence or --no-persistence. Exit')
        return
    
    if args.plots and args.plots != 'show':
        assert os.path.isdir(args.plots)
    
    batch_size = args.batch_size
    if args.task_type == 'upstream':
        modes = ['upstream']
    else:
        modes = [
                'downstream_transfer_mask_and_weights',
                'downstream_transfer_mask_random_weights', 
                'downstream_random_mask_and_weights',
                ]
        if args.few_shot:
            batch_size = min(args.few_shot_size, args.batch_size)
    
    grid = {
        'batch_size': batch_size,
        'n_epochs': args.epochs,
        'model_lr': args.model_lr,
        'mask_lr': args.mask_lr,
        'model_decay': args.model_decay,
        'mask_resting_point': args.mask_resting_point,
        'mask_reg_factor': args.mask_reg_factor,
        'late_mask_training': args.late_mask_training,
        'mode': modes,
        'task': args.task,
        'sparsity': eval(args.sparsity),
        'seed': eval(args.seed),
    }
    
    
    def run_experiment(batch_size, n_epochs, model_lr, mask_lr, model_decay, mask_resting_point,
                       mask_reg_factor, late_mask_training, sparsity, seed, mode, task, config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(seed)
        
        X_train, X_test, model_outputs = get_datasets(task, mode == 'upstream',
                                                      args.few_shot_size if args.few_shot else None)
        
        train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, 
                                                   num_workers=1, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=1, pin_memory=True)
        
        if args.model == 'LogReg':
            inner_model = LogReg(outputs=model_outputs)
        elif args.model == 'Conv1':
            inner_model = Conv1(outputs=model_outputs)
        elif args.model == 'Conv2':
            inner_model = Conv2(outputs=model_outputs)
        elif args.model == 'Conv3':
            inner_model = Conv3(outputs=model_outputs)
        elif args.model == 'ResNet':
            inner_model = resnet18(pretrained=False)
            inner_model.fc = nn.Linear(inner_model.fc.in_features, model_outputs)
            inner_model = WrapperNet(inner_model)
        
        if args.model in ['Conv1', 'Conv2', 'Conv3', 'ResNet']:
            unmaskable_parameters = ['fc.weight', 'fc.bias']
        else:
            assert args.model == 'LogReg'
            unmaskable_parameters = []
        
        masker_training_parameters = MaskerTrainingParameters(
            model_lr=model_lr,
            mask_lr=mask_lr,
            model_l2_decay=model_decay,
            mask_resting_point=mask_resting_point,
            mask_reg_factor=mask_reg_factor,
            warmup_inactive_weights=True
        )
        train_mask = mode == 'upstream' and sparsity > 0
        model = ModelMasker(inner_model, device, masker_training_parameters=masker_training_parameters,
                            sparsity=sparsity, train_mask=train_mask,
                            unmaskable_parameters=unmaskable_parameters)
        best_model = None
        best_dev_score = -np.inf
        
        if mode != 'upstream':
            with Persistence(args.persistence) as db:
                config_first_task = copy.deepcopy(config)
                del config_first_task['batch_size'], config_first_task['n_epochs']
                config_first_task['mode'] = 'upstream'
                entries = db.get_entries(config_first_task)
            assert len(entries) == 1, len(entries)
        
            if mode in ['downstream_transfer_mask_and_weights', 'downstream_transfer_mask_random_weights']:
                if mode == 'downstream_transfer_mask_random_weights':
                    entries[0]['model_state_dict']['model_weights'] = model.model_weights
                model.load_state_dict(entries[0]['model_state_dict'])
                logging.info('Successfully loaded model from upstream task')
            
            elif mode == 'downstream_random_mask_and_weights':
                mask_n_ones = entries[0]['mask_n_ones']
                logging.info('Random mask with %d ones', mask_n_ones)
                model.shift_mask_weights(mask_n_ones)
            else:
                assert False

        train_losses = []
        train_scores = []
        test_losses = []
        test_scores = []
        
        model_optim = optim.RMSprop(model.get_optimizable_parameters(is_model=True), lr=model_lr)
        mask_optim = optim.RMSprop(model.get_optimizable_parameters(is_model=False), lr=mask_lr)
        mask_lr_scheduler = None
        if late_mask_training:
            total_iters = len(train_loader) * 2 # 2 epochs
            mask_lr_scheduler = optim.lr_scheduler.LambdaLR(mask_optim, lambda x: min(x/total_iters, 1))
        
        for epoch in range(n_epochs):
            
            train_loss, train_score = train(args, model, model_optim, mask_optim, None, 
                                            mask_lr_scheduler, device, train_loader, epoch,
                                            verbose=args.verbose, progress_bar=args.progress_bar)
            train_losses.append(train_loss)
            train_scores.append(train_score)
            
            if True: #not args.few_shot or (epoch+1)%10 == 0:
                test_loss, test_score = test(model, device, test_loader, verbose=args.verbose,
                                             progress_bar=args.progress_bar)
                test_losses.append(test_loss)
                test_scores.append(test_score)
                
                if test_score > best_dev_score and model.count_ones()[1] - 0.005 <= 1 - sparsity:
                    best_dev_score = test_score
                    best_model = copy.deepcopy(model).cpu()
                    print('Best model so far')
                
                if args.plots is not None:
                    def show(name):
                        if args.plots == 'show':
                            plt.show()
                        else:
                            plt.savefig('{}/{}_{}.png'.format(args.plots, name, epoch + 1))
                            plt.clf()
                    
                    epoch_len = len(train_loader)
                    epoch_milestones = list(range(0, model.steps + 1, epoch_len))
                    epoch_labels = list(range(0, len(epoch_milestones)))
                    
                    all_mw = [pp.flatten() for pp in model.mask_weights.values()]
                    all_mw = torch.cat(all_mw, 0)
                    plt.hist(all_mw.detach().cpu().numpy())
                    plt.axvline(x=0, c='k')
                    show('mask_weights_hist')
                    
                    all_ltnz = [pp.flatten() for pp in model.last_time_nonzero.values()]
                    all_ltnz = torch.cat(all_ltnz, 0)
                    plt.hist(all_ltnz.cpu().numpy())
                    plt.axhline(y=all_ltnz.numel() * (1 - float(args.sparsity)), c='k')
                    show('last_time_nonzero_hist')
                    
                    # Target density.
                    plt.axhline(y=1 - float(args.sparsity), c='k', linestyle='--', label='Target')
                    # Alive overall density.
                    sorted_ltnz = np.sort(all_ltnz.cpu().numpy())
                    last_index = np.searchsorted(sorted_ltnz, sorted_ltnz[-1], side='left')
                    sorted_ltnz = sorted_ltnz[:last_index]
                    y_values = list(range(all_ltnz.numel(), all_ltnz.numel() - len(sorted_ltnz), -1))
                    y_values = np.array(y_values) / all_ltnz.numel()
                    plt.plot(sorted_ltnz, y_values, label='overall alive', c='C0', linestyle='--')
                    # Overall density.
                    plt.plot(model.densities['overall'], label='overall')
                    # Per-layer density.
                    for pn, pd in model.densities.items():
                        if pn != 'overall':
                            plt.plot(pd, label=pn)
                    plt.xticks(epoch_milestones, labels=epoch_labels)
                    plt.legend()
                    plt.ylim([-0.05, 1.05])
                    show('densities')
                    
                    plt.plot(model.reg_ratios, label='Reg ratios')
                    plt.xticks(epoch_milestones, labels=epoch_labels)
                    plt.legend()
                    show('reg_ratios')
        
        ret = {
            'test_score': best_dev_score,
            'train_losses': train_losses,
            'train_scores': train_scores,
            'test_losses': test_losses,
            'test_scores': test_scores,
            'model_state_dict': best_model.state_dict() if best_model else None,
            'mask_n_ones': int(best_model.count_ones()[0]) if best_model else None
        }
        if return_model:
            ret['return_model'] = model
        return ret
    
    return grid_run(run_experiment, grid, args.persistence, ignore_previous_results=False)


if __name__ == '__main__':
    main()
