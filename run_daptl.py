from __future__ import print_function
import argparse
import logging
import copy
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18

from ModelMasker import ModelMasker, MaskerTrainingParameters
from Nets import LogReg, Net, WrapperNet
from Tasks import get_datasets

from GridRun import grid_run
from Persistence import Persistence


def train(args, model, optimizer, device, train_loader, epoch, lr_factor, verbose=True):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, disable=not verbose)):
        data, target = data.to(device), target.to(device)
        model.zero_grad() # optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target) # Accepts log-probabilities.
        loss.backward()
        train_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        model.sgd_step() # optimizer.step()
        optimizer.step()
        ones, density = model.count_ones()
        if verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDensity: {:.6f} ({})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), density, int(ones)))
    
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
    


def main(args=None):
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
    parser.add_argument('--mask-decay', type=float, default=5e-3, metavar='Maks weight decay',
                        help='L2 and model weight decay (default: 5e-3)')
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
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--plots', action='store_true', default=False)
    parser.add_argument('--few-shot-size', type=int, default=10)
    parser.add_argument('--task', nargs='+', choices=['mnist1','mnist2','mnist3','cifar'], required=True)
    parser.add_argument('--model', choices=['LogReg', 'Conv', 'ResNet'], default='LogReg')
    args = parser.parse_args(args=args)
    
    if args.persistence is None:
        ans = input('You haven\'t specified a persistence. Do you want to continue? (Yes/no): ')
        if ans.lower() == 'no':
            print('Exit')
            exit()
    
    if args.task_type == 'upstream':
        modes = ['upstream']
        batch_size = args.batch_size
    else:
        modes = [
                'downstream_transfer_mask_and_weights',
                'downstream_transfer_mask_random_weights', 
                'downstream_random_mask_and_weights',
                ]
        batch_size = min(args.few_shot_size, args.batch_size)
    
    grid = {
        'batch_size': batch_size,
        'test_batch_size': args.test_batch_size,
        'n_epochs': args.epochs,
        'model_lr': args.model_lr,
        'mask_lr': args.mask_lr,
        'model_decay': args.model_decay,
        'mask_decay': args.mask_decay,
        'lr_factor': 0.9,
        'mode': modes,
        'task': args.task,
        'sparsity': eval(args.sparsity),
        'seed': eval(args.seed),
    }
    
    
    def run_experiment(batch_size, test_batch_size, n_epochs, model_lr, mask_lr, model_decay, mask_decay, sparsity,
                    lr_factor, seed, mode, task, config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(seed)
        
        X_train, X_test, model_outputs = get_datasets(task, mode == 'upstream', args.few_shot_size)
        
        train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(X_test, batch_size=test_batch_size, shuffle=False)
        
        if args.model == 'LogReg':
            inner_model = LogReg(outputs=model_outputs)
        elif args.model == 'Conv':
            inner_model = Net(outputs=model_outputs)
        elif args.model == 'ResNet':
            inner_model = WrapperNet(resnet18(pretrained=True), outputs=model_outputs)
            
            # Fix model output and input sizes.
            
        masker_training_parameters = MaskerTrainingParameters(
            model_lr=model_lr,
            mask_lr=mask_lr,
            model_l2_decay=model_decay,
            mask_l2_decay=mask_decay,
            mask_sl1_decay=mask_decay,
            mask_grad_eps=0
        )
        model = ModelMasker(inner_model, device, masker_training_parameters=masker_training_parameters,
                            sparsity=sparsity)
        
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
        
        optimizer = optim.RMSprop([
            {'params': model.get_optimizable_parameters(is_model=True), 'lr': model_lr},
            {'params': model.get_optimizable_parameters(is_model=False), 'lr': mask_lr}
        ])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_factor)
        
        for epoch in range(n_epochs):
            
            train_loss, train_score = train(args, model, optimizer, device, train_loader, epoch, lr_factor, verbose=args.verbose)
            lr_scheduler.step()
            train_losses.append(train_loss)
            train_scores.append(train_score)
            
            if True: #not args.few_shot or (epoch+1)%10 == 0:
                test_loss, test_score = test(model, device, test_loader, verbose=args.verbose)
                test_losses.append(test_loss)
                test_scores.append(test_score)
                
                if args.plots:
                    all_mw = [pp.flatten() for pp in model.mask_weights.values()]
                    all_mw = torch.cat(all_mw, 0)
                    plt.hist(all_mw.detach().cpu().numpy())
                    plt.axvline(x=0, c='k')
                    plt.show()
                    print(all_mw.min())

                    all_ltnz = [pp.flatten() for pp in model.last_time_nonzero.values()]
                    all_ltnz = torch.cat(all_ltnz, 0)
                    plt.hist(all_ltnz.cpu().numpy())
                    plt.axhline(y=all_ltnz.numel() * (1 - float(args.sparsity)), c='k')
                    plt.show()
        
        #plt.imshow(model.model.fc1.weight.data.cpu().numpy()[0].reshape(8,8))
        #plt.show()

        return {
            'test_score': test_scores[-1],
            'train_losses': train_losses,
            'train_scores': train_scores,
            'test_losses': test_losses,
            'test_scores': test_scores,
            'model_state_dict': model.state_dict(),
            'mask_n_ones': int(model.count_ones()[0])
        }
    
    grid_run(run_experiment, grid, args.persistence, ignore_previous_results=True)


if __name__ == '__main__':
    main()
