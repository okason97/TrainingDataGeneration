import torch
import torch.nn as nn
import copy
import numpy as np
from math import inf
from torchvision import transforms
import sys
sys.path.append('./')
from misc import mixup_data, mixup_criterion, sigmoid

def train(model, dataloaders, epochs, mode, device):

    print("Training started")

    # -----------------------------------------------------------------------------
    # mixup parameters used to calculate the beta distribution.   
    # -----------------------------------------------------------------------------
    alpha = 1.0
    beta = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)

    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = inf

    if mode == 'gen':
        gen_dataloader = {'train': dataloaders['gen'],
                          'val': dataloaders['val']}
        model, _ = train_loop(model, gen_dataloader, epochs, mode, device)
        epochs = int(epochs/4)

    model, results = train_loop(model, dataloaders, epochs, mode, device)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results

def train_loop(model, dataloaders, epochs, mode, device):
    confidence_alpha = -3.
    confidence_beta = 0.5
    confidence_max = 1.

    results = {
        'train': {
            'epoch_loss': np.zeros(epochs),
            'epoch_acc': np.zeros(epochs),
        },
        'val': {
            'epoch_loss': np.zeros(epochs),
            'epoch_acc': np.zeros(epochs),
        },
    }

    for epoch in range(epochs):
        print('##############################')
        print('#{} epoch: {}'.format(name, epoch))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if phase == 'train':
                    if mode == 'reggen':
                        inputs_g, labels_g = next(iter(dataloaders['gen']))
                        idx = torch.randperm(inputs.shape[0])
                        inputs_g, lam_g = mixup_data(inputs_g[idx], inputs_g, alpha, beta)
                        labels_ga = labels_g[idx].to(device)
                        labels_gb = labels_g.to(device)
                    if mode == 'mixgen':
                        inputs_g, labels_g = next(iter(dataloaders['gen']))
                        inputs, lam = mixup_data(inputs_g, inputs, alpha, beta)
                        labels_a = labels_g.to(device)
                        labels_b = labels.to(device)
                    else:
                        idx = torch.randperm(inputs.shape[0])
                        inputs, lam = mixup_data(inputs[idx], inputs, alpha, beta)
                        labels_a = labels[idx].to(device)
                        labels_b = labels.to(device)
                else:
                    inputs = inputs.to(device)                    
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if mode == 'reggen':
                        outputs_g = model(inputs_g)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        if mode == 'reggen':
                            loss_g = mixup_criterion(criterion, outputs_g, labels_ga, labels_gb, lam_g)
                            loss = loss + (1-epoch/epochs)*loss_g
                    else:
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    running_corrects += torch.sum(torch.logical_or(preds == labels_a.data,preds == labels_b.data))
                else:
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            results[phase]['epoch_loss'][epoch] = epoch_loss
            results[phase]['epoch_acc'][epoch] = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, results[phase]['epoch_loss'][epoch], results[phase]['epoch_acc'][epoch]))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

def test(model, dataloaders, device = 'cuda'):

    print("Testing started")

    results = {
        'test': {
            'loss': None,
            'acc': None,
        },
    }

    criterion = nn.CrossEntropyLoss()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    results['test']['loss'] = running_loss / len(dataloaders['test'].dataset)
    results['test']['acc'] = running_corrects.double() / len(dataloaders['test'].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'Test', results['test']['loss'], results['test']['acc']))

    return results