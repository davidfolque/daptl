import torch
from torchvision import datasets, transforms

import copy

def split_dataset(dataset, class_sizes):
    dataset1 = copy.deepcopy(dataset)
    dataset2 = copy.deepcopy(dataset)
    idx = []
    idx2 = []
    for i in range(10):
        elements = torch.nonzero(dataset.targets == i)
        # We trust that the seed has been initialised before.
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

def get_datasets(task, is_upstream, few_shot_size=None):
    if task in ['mnist1', 'mnist2', 'mnist3']:
        transform=transforms.Compose([
            #transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(size=(8,8))
            ])
        
        if is_upstream or not few_shot_size:
            X_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
            X_test = datasets.MNIST('../data', train=False, download=True, transform=transform)
        else:
            X_all = datasets.MNIST('../data', train=False, download=True, transform=transform)
        
        train_size = 5000
        test_size = 500
        
        def select_samples(classes):
            if is_upstream or not few_shot_size:
                X_train, _ = split_dataset(X_train, [cl * train_size for cl in class_sizes])
                X_test, _ = split_dataset(X_test, [cl * test_size for cl in class_sizes])
            else:
                X_train, X_rest = split_dataset(X_all, [cl * few_shot_size for cl in class_sizes])
                X_test, _ = split_dataset(X_rest, [cl * test_size for cl in class_sizes])
            return X_train, X_test
        
        if task == 'mnist1':
            model_outputs = 2
            if is_upstream:
                # Select only the 0 and 1s.
                X_train, X_test = select_samples(2 * [1] + 8 * [0])
            else:
                # Select only the 2 and 3s.
                X_train, X_test = select_samples(2 * [0] + 2 * [1] + 6 * [0])
                # Update the labels.
                X_train.targets -= 2
                X_test.targets -= 2
        
        elif task == 'mnist2':
            model_outputs = 5
            if is_upstream:
                # Select all 0-4s.
                X_train, X_test = select_samples(5 * [1] + 5 * [0])
            else:
                # Select all 5-9s.
                X_train, X_test = select_samples(5 * [0] + 5 * [1])
                # Update the labels.
                X_train.targets -= 5
                X_test.targets -= 5
        
        elif task == 'mnist3':
            model_outputs = 2
            if is_upstream:
                # Select all numbers.
                X_train, X_test = select_samples(10 * [1])
                # Update labels.
                X_train.targets %= 2
                X_test.targets %= 2
            else:
                # Select all numbers.
                X_train, X_test = select_samples(10 * [1])
                # Update the labels.
                X_train.targets /= 5
                X_test.targets /= 5
                
        else: assert False
        
        return X_train, X_test, model_outputs
    
    elif task in ['cifar-32', 'cifar-224']:
        
        transforms_list = []
        if task == 'cifar-224':
            transforms_list.append(transforms.Resize(224))
        
        transforms_list += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        
        if is_upstream:
            # Computed by us for CIFAR-100:
            transforms_list.append(transforms.Normalize(
                (0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)))
        else:
            # Computed by us for CIFAR-10:
            transforms_list.append(transforms.Normalize(
                (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)))
        ## Normalization specified by pytorch for resnet18 model.
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose(transforms_list)
        
        if is_upstream:
            X_train = datasets.CIFAR100(root='../data', train=True,
                                        download=True, transform=transform)
            X_test = datasets.CIFAR100(root='../data', train=False,
                                       download=True, transform=transform)
            return X_train, X_test, 100
        
        else:
            X_train = datasets.CIFAR10(root='../data', train=True,
                                       download=True, transform=transform)
            X_test = datasets.CIFAR10(root='../data', train=False,
                                      download=True, transform=transform)
            X_train, _ = split_dataset(X_train, 10 * [few_shot_size])
            
            return X_train, X_test, 10
        
        

        
        
        
        
        
