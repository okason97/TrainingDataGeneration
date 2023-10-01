import torch
import os
import torch.nn as nn
import copy
import numpy as np
from math import inf
import sys
import shutil
import pickle
from torchvision.utils import save_image
sys.path.append('./')
from utils.misc import mixup_data, mixup_criterion
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import random
import math

def train(model, dataloaders, epochs, growth, reg_alpha, reg_beta, mode, device):

    print("Training started")

    # -----------------------------------------------------------------------------
    # mixup parameters used to calculate the beta distribution.   
    # -----------------------------------------------------------------------------

    if mode == 'gen':
        gen_dataloader = {'train': dataloaders['gen'],
                          'val': dataloaders['val']}
        model, _, _ = train_loop(model, gen_dataloader, int(epochs/4), growth, reg_alpha, reg_beta, mode, device)

    model, results, best_model_wts = train_loop(model, dataloaders, epochs, growth, reg_alpha, reg_beta, mode, device)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results

def growth_function(x,alpha,beta):
    return alpha+(1-alpha)*(1-math.exp(-beta*x))

def decay_function(x,alpha,beta):
    return alpha*math.exp(-beta*x)

def train_loop(model, dataloaders, epochs, growth, reg_alpha, reg_beta, mode, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999), eps=1e-6)

    criterion = nn.CrossEntropyLoss()

    if growth:
        change_function = growth_function
    else:
        change_function = decay_function

    alpha = 5.0
    beta = 5.0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = inf

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
        print('#epoch: {}'.format(epoch))

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
                        inputs_g = inputs_g.to(device)
                        labels_ga = labels_g[idx].to(device)
                        labels_gb = labels_g.to(device)
                    if mode == 'mixgen' and random.random() >= change_function(epoch,reg_alpha,reg_beta):
                        inputs_g, labels_g = next(iter(dataloaders['gen']))
                        inputs, lam = mixup_data(inputs_g, inputs, alpha, beta)
                        labels_a = labels_g.to(device)
                        labels_b = labels.to(device)
                    else:
                        idx = torch.randperm(inputs.shape[0])
                        inputs, lam = mixup_data(inputs[idx], inputs, alpha, beta)
                        labels_a = labels[idx].to(device)
                        labels_b = labels.to(device)
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
                            loss = loss + loss_g * change_function(epoch,reg_alpha,reg_beta)
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
            epoch_acc = float(running_corrects) / len(dataloaders[phase].dataset)

            results[phase]['epoch_loss'][epoch] = epoch_loss
            results[phase]['epoch_acc'][epoch] = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, results[phase]['epoch_loss'][epoch], results[phase]['epoch_acc'][epoch]))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    return model, results, best_model_wts

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
    results['test']['acc'] = float(running_corrects) / len(dataloaders['test'].dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'Test', results['test']['loss'], results['test']['acc']))

    return results

def filter(model, dataloaders, top_k, n_classes, save_dir, device = 'cuda'):

    print("Filtering best samples")

    model.eval()   # Set model to evaluate mode

    filtered =  np.zeros((n_classes, 2, top_k))

    cur_index = np.zeros(n_classes, dtype=np.int32)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for f in range(n_classes):
        if not os.path.exists(os.path.join(save_dir, str(f))):
            os.makedirs(os.path.join(save_dir, str(f)))

    softmax = nn.Softmax(dim=1)

    # Iterate over data.
    for inputs, labels, indices in dataloaders['gen']:
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        indices = indices.to(device)

        # forward
        outputs = softmax(model(inputs))

        print(len(labels))
        print(len(outputs))

        for i in range(len(outputs)):
            score = outputs[i][labels[i]]
            if cur_index[labels[i]] < top_k:
                filtered[labels[i], 0, cur_index[labels[i]]] = score
                filtered[labels[i], 1, cur_index[labels[i]]] = indices[i]
                cur_index[labels[i]] += 1
            else:
                min_index = np.argmin(filtered[labels[i],0])
                if score < filtered[labels[i], 0, min_index]:
                    filtered[labels[i], 0, min_index] = score
                    filtered[labels[i], 1, min_index] = indices[i]

    scores = np.zeros((n_classes, top_k))
    for i in range(n_classes):
        print('##############################')
        print('#class: {}'.format(i))
        print('Min score: {:.4f}, Max score: {:.4f}'.format(np.min(filtered[i, 0]), np.max(filtered[i, 0])))
        for j in range(top_k):
            save_image(dataloaders['gen'].dataset[int(filtered[i, 1, j])][0].float()/255, os.path.join(save_dir, str(i), "{idx}.png".format(idx=j)))
            scores[i, j] = filtered[i, 0, j]
    with open(save_dir+'scores.pkl', 'wb') as f:
        pickle.dump(scores, f)

def create_confusion_matrix(model, dataloaders, save_dir, device = 'cuda'):

    print("Creating confusion matrix")

    y_true, y_pred = get_preds(model, dataloaders, device)
  
    cfm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(cfm, annot=False)
    cfm_plot.figure.savefig(save_dir+"cfm.png")

def calculate_class_accuracy(model, dataloaders, save_dir, device = 'cuda'):
    print("Calculating class accuracy")

    y_true, y_pred = get_preds(model, dataloaders, device)
   
    ca = class_accuracy(y_true, y_pred)
    with open(save_dir+'class_accuracy.pkl', 'wb') as f:
        pickle.dump(ca, f)

def class_accuracy(y_true, y_pred):
    class_accuracy = {}

    for y_t, y_p in zip(y_true, y_pred):
        if y_t not in class_accuracy:
            class_accuracy[y_t] = {"t_sum": y_t==y_p, "count": 1}
        else:
            class_accuracy[y_t]["t_sum"] += (y_t==y_p)            
            class_accuracy[y_t]["count"] += 1            

    for key in class_accuracy:
        class_accuracy[key] = class_accuracy[key]["t_sum"]/class_accuracy[key]["count"]
    
    return class_accuracy

def get_preds(model, dataloaders, device = 'cuda'):
    model.eval()   # Set model to evaluate mode

    y_true = []
    y_pred = []

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true += labels.tolist()
        y_pred += preds.tolist()

    return y_true, y_pred