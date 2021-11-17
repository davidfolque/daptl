from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging

import copy
from ModelMasker import ModelMasker
from Nets import LogReg

from GridRun import grid_run
from Persistence import Persistence

# Train mask and model on 0-4.
# Freeze mask and re-train model on 5-9.


def train(args, model, device, train_loader, epoch, lr_factor, verbose=True):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.zero_grad() # optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        model.sgd_step() # optimizer.step()
        ones, density = model.count_ones()
        if verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDensity: {:.6f} ({})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), density, int(ones)))
    # Scheduler step.
    model.model_lr *= lr_factor
    model.mask_lr *= lr_factor
    
    # Loss, accuracy
    return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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
    


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model_lr', type=float, default=0.05, metavar='Model LR',
                        help='model learning rate (default: 0.05)')
    parser.add_argument('--mask_lr', type=float, default=1., metavar='Mask LR',
                        help='mask learning rate (default: 1.0)')
    parser.add_argument('--decay', type=float, default=1e-4, metavar='Weight decay',
                        help='L2 and signed-L1 decay (default: 1e-4)')
    parser.add_argument('--lr_factor', type=float, default=0.9, metavar='M',
                        help='Learning rate scheduler factor (default: 0.9)')
    #parser.add_argument('--seed', type=int, default=1, metavar='S',
                        #help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    #parser.add_argument('--random-mask-sparsity', type=int, default=0,
                        #help='Number of ones in generated random mask')
    parser.add_argument('--task-type', choices=['upstream', 'downstream'], required=True)
    parser.add_argument('--persistence', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--few-shot', action='store_true', default=False)
    parser.add_argument('--few-shot-size', type=int, default=10)
    args = parser.parse_args()
    
    
    if args.task_type == 'upstream':
        modes = ['upstream']
    else:
        modes = [
                'downstream_transfer_mask_and_weights',
                'downstream_transfer_mask_random_weights', 
                'downstream_random_mask_and_weights',
                ]
    
    if args.few_shot:
        train_size = args.few_shot_size
        batch_size = 10
    else:
        train_size = 5000
        batch_size = args.batch_size
    
    grid = {
        'batch_size': batch_size,
        'test_batch_size': args.test_batch_size,
        'n_epochs': args.epochs,
        'model_lr': args.model_lr,
        'mask_lr': args.mask_lr,
        'decay': args.decay,
        'lr_factor': 0.9,
        'mode': modes,
        'tasks': ['0-1/2-3', '0-4/5-9', '%2/<5'],
        'seed': [1, 2, 3, 4, 5],
    }
    
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize(size=(8,8))
        ])
    
    def split_dataset(dataset, class_sizes):
        dataset1 = copy.deepcopy(dataset)
        dataset2 = copy.deepcopy(dataset)
        idx = []
        idx2 = []
        for i in range(10):
            elements = torch.nonzero(dataset.targets == i)
            perm = elements[torch.randperm(len(elements))]
            idx += perm[:class_sizes[i]]
            idx2 += perm[class_sizes[i]:]
        idx = torch.cat(idx, 0)
        idx2 = torch.cat(idx2, 0)
        
        dataset1.targets = dataset.targets[idx]
        dataset1.data = dataset.data[idx]
        dataset2.targets = dataset.targets[idx2]
        dataset2.data = dataset.data[idx2]
        return dataset1, dataset2
    
    
    def run_experiment(batch_size, test_batch_size, n_epochs, model_lr, mask_lr, decay,
                    lr_factor, seed, mode, tasks, config):
        device = 'cpu'
        torch.manual_seed(seed)
        
        X_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
        X_test = datasets.MNIST('../data', train=False, transform=transform)
        
        if mode == 'upstream':
            if tasks == '0-1/2-3':
                model_outputs = 2
                X_train = split_dataset(X_train, 2 * [train_size] + 8 * [0])[0]
                X_test = split_dataset(X_test, 2 * [500] + 8 * [0])[0]
            elif tasks == '0-4/5-9':
                model_outputs = 5
                X_train = split_dataset(X_train, 5 * [train_size] + 5 * [0])[0]
                X_test = split_dataset(X_test, 5 * [500] + 5 * [0])[0]
            elif tasks == '%2/<5':
                model_outputs = 2
                X_train = split_dataset(X_train, 10 * [train_size])[0]
                X_test = split_dataset(X_test, 10 * [500])[0]
                X_train.targets %= 2
                X_test.targets %= 2
            else:
                assert False
        else:
            assert mode in ['downstream_transfer_mask_and_weights', 'downstream_random_mask_and_weights',
                            'downstream_transfer_mask_random_weights']
            
            def select_elements(train_distr, test_distr):
                if args.few_shot:
                    train, test = split_dataset(X_test, train_distr)
                    test = split_dataset(test, test_distr)[0]
                else:
                    train = split_dataset(X_train, train_distr)[0]
                    test = split_dataset(X_test, test_distr)[0]
                return train, test
            
            if tasks == '0-1/2-3':
                model_outputs = 2
                X_train, X_test = select_elements(2 * [0] + 2 * [train_size] + 6 * [0], 
                                                  2 * [0] + 2 * [500] + 6 * [0])
                X_train.targets -= 2
                X_test.targets -= 2
            elif tasks == '0-4/5-9':
                model_outputs = 5
                X_train, X_test = select_elements(5 * [0] + 5 * [train_size],
                                                  5 * [0] + 5 * [500])
                X_train.targets -= 5
                X_test.targets -= 5
            elif tasks == '%2/<5':
                model_outputs = 2
                X_train, X_test = select_elements(10 * [train_size],
                                                  10 * [500])
                X_train.targets //= 5
                X_test.targets //= 5
            else:
                assert False
    
        
        train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=test_batch_size, shuffle=False)
        
        model = ModelMasker(LogReg(outputs=model_outputs), device, model_lr=model_lr, mask_lr=mask_lr, decay=decay)
        
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
        
        model.training_mask = mode == 'upstream'

        train_losses = []
        train_scores = []
        test_losses = []
        test_scores = []
        
        for epoch in range(n_epochs):
            
            train_loss, train_score = train(args, model, device, train_loader, epoch, lr_factor, verbose=args.verbose)
            train_losses.append(train_loss)
            train_scores.append(train_score)
            
            if not args.few_shot or (epoch+1)%10 == 0:
                test_loss, test_score = test(model, device, test_loader, verbose=args.verbose)
                test_losses.append(test_loss)
                test_scores.append(test_score)
        
        return {
            'test_score': test_scores[-1],
            'train_losses': train_losses,
            'train_scores': train_scores,
            'test_losses': test_losses,
            'test_scores': test_scores,
            'model_state_dict': model.state_dict(),
            'mask_n_ones': int(model.count_ones()[0])
        }
    
    grid_run(run_experiment, grid, args.persistence, ignore_previous_results=False)


if __name__ == '__main__':
    main()
