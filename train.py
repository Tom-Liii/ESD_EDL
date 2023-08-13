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

def train_model(
        args, 
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
):

    since = time.time()

    if not device:
        device = get_device()
    # print(device)
    logger = logging.getLogger("my logger") 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_dice = 0.0
    losses = {"train": [], 'val':[]}
    accuracy = {"train": [], "val":[]}
    eces = {'train':[], 'val':[]}
    dice_criterion = DiceLoss(
            mode='multiclass',
            classes=num_classes,
            log_loss=False,
            from_logits=True,
            smooth=0.001,
            ignore_index=None,
        )
    dice_metric = DiceMetric()
    

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        
        # Iterate over data.
        model.train()
        print('Training phase')

        phase = 'train'
        running_loss = 0.0
        running_corrects = 0.0
        running_ece = 0.0
        running_dice = 0.0
        running_mi = 0.0
        for i, (inputs, labels) in enumerate(dataloaders[phase]):

            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.shape)
            optimizer.zero_grad()

            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                # print(y.shape)
                outputs = model(inputs)
                # print(inputs.shape, outputs.shape)
                _, preds = torch.max(outputs, 1)
                loss = criterion(
                    outputs, y.float(), epoch, num_classes, 10, device
                )

                # print(policy_gradient_loss(outputs, labels))

                # match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                match = torch.eq(preds, labels).float()
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                if epoch == 0 and i == 0:
                    print('preds shape:', preds.shape)
                    print('u shape:', u.shape)
                    print('u slice len:', len(u[0, 0, 0:, 0]))
                    print('u slice max:', max(u[0, 0, 0:, 0]))
                    print('u slice:', u[0, 0, 0:, 0])
                expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
                
                ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
                dice = dice_metric.dice_coef(expected_prob, labels)
                mi = calc_mi(outputs, labels)
                
                
                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

            else:
                outputs = model(inputs)
                softmax_pred = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                loss = dice_criterion(outputs, labels)
                # print(loss, inputs.shape, outputs.shape, labels.shape)
                # print(dice_criterion(outputs, labels))
                ece = calc_ece_softmax(softmax_pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
                # dice = _eval_dice(labels.cpu(), preds.detach().cpu(), num_classes)
                dice = dice_metric.dice_coef(softmax_pred, labels)
                mi = calc_mi(outputs, labels)

            
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data) / (target_img_size * target_img_size)
            running_ece += ece * inputs.size(0) 
            running_dice += dice * inputs.size(0) 
            running_mi += mi * inputs.size(0)

        # print(evaluation(model, dataloaders['val'], [], []))
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_dice = running_dice / len(dataloaders[phase].dataset)
        epoch_ece = running_ece / len(dataloaders[phase].dataset)
        epoch_mi = running_mi / len(dataloaders[phase].dataset)


        losses[phase].append(epoch_loss)
        accuracy[phase].append(epoch_acc.item())
       

        logger.info(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece: {:.4f} mi: {:.4f}".format(
                'Train', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )

        print(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece: {:.4f} mi: {:.4f}".format(
                'Train', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )
        


        model.eval()
        print('Evaluation phase')
        phase = 'val'
        running_loss = 0.0
        running_corrects = 0.0
        running_ece = 0.0
        running_dice = 0.0
        running_mi = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                if uncertainty:
                    y = one_hot_embedding(labels, num_classes)
                    y = y.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print('___________')
                    # print('preds: ')
                    # print(preds)
                    # print('labels: ')
                    # print(labels)
                    # print('___________')
                    loss = criterion(
                        outputs, y.float(), epoch, num_classes, 10, device
                    )

                    match = torch.eq(preds, labels).float()
                    acc = torch.mean(match)
                    evidence = relu_evidence(outputs)
                    alpha = evidence + 1

                    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                    # print(u.shape)
                    # all_uncertainty.extend(list(u.squeeze().detach().cpu().numpy()))
                    # print(all_uncertainty)
                    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
                
                    ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    dice = dice_metric.dice_coef(expected_prob, labels)
                    mi = calc_mi(outputs, labels)

                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match
                    ) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)
                    ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    softmax_pred = F.softmax(outputs, dim=1)

                    # loss = criterion(outputs, labels)
                    loss = dice_criterion(outputs, labels)
                    # dice = _eval_dice(labels.cpu(), preds.detach().cpu(), num_classes)
                    ece = calc_ece_softmax(softmax_pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    # dice = 1 - loss.item()
                    dice = dice_metric.dice_coef(softmax_pred, labels)
                    mi = calc_mi(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (torch.sum(preds == labels.data)) / (target_img_size * target_img_size)
                running_ece += ece * inputs.size(0)
                running_dice += dice * inputs.size(0)
                running_mi += mi * inputs.size(0)

       

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_ece = running_ece / len(dataloaders[phase].dataset)
        epoch_dice = running_dice / len(dataloaders[phase].dataset)
        epoch_mi = running_mi / len(dataloaders[phase].dataset)
        # print(len(dataloaders[phase].dataset))
        # print(len(dataloaders[phase].dataset))
        # print(sum(all_uncertainty)/len(all_uncertainty))
        # if epoch % 20 ==0 :
        #     sns.kdeplot(np.array(all_uncertainty))
            
        #     plt.show()
        losses[phase].append(epoch_loss)
        accuracy[phase].append(epoch_acc.item())

        if phase == "val" and epoch_dice > best_dice:    
            best_dice = epoch_dice
            best_ece = epoch_ece
            best_mi = epoch_mi
            best_model_wts = copy.deepcopy(model.state_dict())
            save(args, model, optimizer)
            # save_best(model, optimizer)
            # save_final(model, optimizer)
        
        logger.info(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece {:.4f} mi: {:.4f}".format(
                'Test', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )
        print(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece {:.4f} mi: {:.4f}".format(
                'Test', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )
        logger.info('-------------------------------------------')



    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # print("Best val Acc: {:4f}".format(best_acc))
    logger.info(
            "Best model dice: {:.4f} ece: {:.4f} mi: {:.4f}".format(
              best_dice, best_ece, best_mi
            )
        )

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
