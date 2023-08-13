import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import random
import argparse
import logging
import sys
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
# from data import dataloaders, digit_one
from train import train_model
from train_esd import test_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet
from segmentation_models_pytorch.unet.model import Unet
from esd_dataset import ESD_Dataset, get_split

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train", type=int, default=1, help="To train the network."
    )
    parser.add_argument("--test", action="store_true", help="To test the network.")
    parser.add_argument(
        "--examples", action="store_true", help="To example MNIST data."
    )
    parser.add_argument(
        "--epochs", default=500, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--train_batch_size", default=4, type=int, help="Desired number of train batch size."
    )
    parser.add_argument(
        "--val_batch_size", default=1, type=int, help="Desired number of val batch size."
    )
    parser.add_argument(
        "--num_classes", default=5, type=int, help="Desired number of classes."
    )
    parser.add_argument(
        "--dropout", action="store_true", help="Whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", type=int, default=0, help="Use uncertainty or not."
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Use uncertainty or not."
    )
    
    parser.add_argument(
        "--mse",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    parser.add_argument(
        "--digamma",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    parser.add_argument(
        "--log",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )
    parser.add_argument(
        "--only_test",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )

    # parser.add_argument(
    #     "--test",
    #     default=0, type=int,
    #     help="Set this argument when testing. ",
    # )
    args = parser.parse_args()
    
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    
    seed = args.seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    log_file_name = 'log'
    log_file_name += '_seed_' + str(seed)
    log_file_name += '_batch_' + str(train_batch_size)
    log_file_name += '_classes_' + str(args.num_classes)
    if args.uncertainty :
        log_file_name += '_edl'
        if args.mse:
            log_file_name += '_mse'
        elif args.digamma:
            log_file_name += '_digamma'
        elif args.log:
            log_file_name += '_log'
    # log_file_name = 'debug'
    logging.basicConfig(filename=log_file_name, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('My logger').addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    
    if args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = args.num_classes

        train_ids, val_ids, test_ids = get_split()
        # print('___________')
        # print(test_ids[0])
        # print('___________')
        train_dataset = ESD_Dataset(train_ids)
        val_dataset = ESD_Dataset(val_ids)
        test_dataset = ESD_Dataset(test_ids, ids=True)
        # train_dataset = ESD_Dataset(train_ids, ids=True)
        # val_dataset = ESD_Dataset(val_ids, ids=True)
        # test_dataset = ESD_Dataset(test_ids, ids=True)
        # print(train_dataset[0])

        # improve the reusability of the dataset
        # train_dataset = torch.load('./data/train_dataset.pt')
        # val_dataset = torch.load('./data/val_dataset.pt')
        # test_dataset = torch.load('./data/test_dataset.pt')

        dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        dataloader_val = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)
        dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)
        # inputs, _ = next(iter(dataloader_train))
        # inputs, _, _ = next(iter(dataloader_train))
        # input_dims = inputs.shape[1:]
        # print(input_dims) # torch.Size([3, 256, 256])
        # exit()
        dataloaders = {
            "train": dataloader_train,
            "val": dataloader_val,
            'test': dataloader_test,
        }

    

        model = Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=3,
            classes=num_classes,
        )

        if use_uncertainty:
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)
        # print('begin to train')
        # print(f' args: {args}\n model: {model}\n dataloaders: {dataloaders}\n num_classes: {num_classes}\n criterion: {criterion}\n optimizer: {optimizer}\n num_epochs: {num_epochs}\n device: {device}\n uncertainty: {use_uncertainty}')
        if args.only_test:
            from pathlib import Path
            model_path = Path('./results/model_uncertainty_mse_batch_4_classes_5_seed_2')
            # Load the checkpoint
            checkpoint = torch.load(model_path)

            # Load the model state dict
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load the optimizer state dict
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            model.cuda()
            device = get_device()
            model = model.to(device)
            print('begin to test')
            predictions, images, labels_list, ids_list, u_list = test_model(
                model,
                dataloaders
            )
            from evaluation_utils import save_segmentation_images
            save_segmentation_images(images, labels_list, predictions, '/research/d5/gds/hzyang22/Heming_Li/uncertainty/EDL/results/figure4', ids_list, u_list)
            return 
        
        model, metrics = train_model(
        # test_model(
            args, 
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=None,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )
        
        # state = {
        #     "epoch": num_epochs,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     'metrics': metrics,
        # }

        # if use_uncertainty:
        #     if args.digamma:
        #         checkpoint_name = "./results/model_uncertainty_digamma" + '_batch_' + str(train_batch_size)
        #         torch.save(state, checkpoint_name)
                
        #     if args.log:
        #         checkpoint_name = "./results/model_uncertainty_log" + '_batch_' + str(train_batch_size)
        #         torch.save(state, checkpoint_name)
                
        #     if args.mse:
        #         checkpoint_name = "./results/model_uncertainty_mse" + '_batch_' + str(train_batch_size)
        #         torch.save(state, checkpoint_name)
                

        # else:
        #     checkpoint_name = "./results/model" + '_batch_' + str(train_batch_size)
        #     torch.save(state, checkpoint_name)
            

    


if __name__ == "__main__":
    main()
