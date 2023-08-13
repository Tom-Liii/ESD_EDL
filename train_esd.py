import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import get_device, one_hot_embedding
from losses import relu_evidence, policy_gradient_loss
from metrics import calc_ece_evidence_u, calc_ece_softmax, DiceMetric, calc_mi
from segmentation_models_pytorch.losses.dice import DiceLoss
from esd_dataset import target_img_size
from rl_tuning import evaluation

device = get_device()

def save(args, model, optimizer):
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

    if args.uncertainty:
        if args.digamma:
            checkpoint_name = "./results/model_uncertainty_digamma" 
            
        if args.log:
            checkpoint_name = "./results/model_uncertainty_log"
            
        if args.mse:
            checkpoint_name = "./results/model_uncertainty_mse" 
    else:
        checkpoint_name = "./results/model" 
    checkpoint_name += '_batch_' + str(args.train_batch_size) + '_classes_' + str(args.num_classes) + '_seed_' + str(args.seed)
    torch.save(state, checkpoint_name)

def save_best(model, optimizer): 
    name = './results/model/best_model.pt'
    torch.save(model.state_dict(), name)

def save_final(model, optimizer): 
    name = './results/model/final_model.pt'
    torch.save(model.state_dict(), name)

def test_model(
        model,
        dataloaders,
        uncertainty=True,
        num_classes=5
):

    since = time.time()

    device = get_device()
    
    



    model.eval()
    print('Evaluation phase')
    phase = 'test'
    running_loss = 0.0
    running_corrects = 0.0
    running_ece = 0.0
    running_dice = 0.0
    running_mi = 0.0
    with torch.no_grad():
        predictions = []
        images, labels_list = [], []
        ids_list = []
        u_list = []
        for i, (inputs, labels, ids) in enumerate(dataloaders[phase]):

            inputs = inputs.to(device)
            labels = labels.to(device)
            ids_list += ids

            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                

                match = torch.eq(preds, labels).float()
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                predictions.append(outputs.cpu().numpy())
                images.append(inputs.cpu().numpy())  # (B, H, W)
                labels_list.append(labels.cpu().numpy())  # (B, H, W)
                u_list.append(u.cpu().numpy())
                # print('u shape:', u.shape)
                if i == 0: 
                    print('u len:', len(u[0, 0, 0:, 0]))
                    print('u max:', max(u[0, 0, 0:, 0]))
                    print('u:', u[0, 0, 0:, 0])

                
    predictions = np.concatenate(predictions, axis=0)
    images = np.concatenate(images, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    u_list = np.concatenate(u_list, axis=0)
    print(predictions.shape, images.shape, labels_list.shape, u_list.shape)
    print('__________________')
    return predictions, images, labels_list, ids_list, u_list

    
