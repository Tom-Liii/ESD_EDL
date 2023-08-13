import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize
)
from torchvision.transforms import ToTensor
from torchvision import transforms

target_img_size = 256
# def image_transform(p=1):
#     return Compose([
#         Resize(target_img_size, target_img_size, cv2.INTER_LINEAR),
#         # Normalize(p=1)
#     ], p=p)


# def mask_transform(p=1):
#     return Compose([
#         Resize(target_img_size, target_img_size, cv2.INTER_NEAREST)
#     ], p=p)

def image_transform(p=1):
    return Compose([
       Resize(448, 512, cv2.INTER_LINEAR),
        Normalize(p=1)
    ], p=p)


def mask_transform(p=1):
    return Compose([
       Resize(448, 512, cv2.INTER_NEAREST)
    ], p=p)

def get_id(input_img_path):
    data_path = Path(input_img_path) # added
    pred_file_name = []
    pred_file_name.append(data_path)
    return pred_file_name

def get_split():
    # data_path = Path('/research/d5/gds/hzyang22/data/ESD_organized')
    dataset_path = Path('/research/d5/gds/hzyang22/data/new_esd_seg') # original
    # data_path = Path('C:/Users/student/Desktop/EDL/new_esd_seg')  
    
    # dataset_path = Path('C:/Users/tom/Desktop/summer_research/new_esd_seg') # added
# Set the path to the dataset folder

    # Initialize the lists
    train_data_file_names = []
    val_data_file_names = []
    test_data_file_names = []

    # Iterate over the subdirectories
    for subdir in os.listdir(dataset_path):
        sub_dir_path = os.path.join(dataset_path, subdir)
        sub_dir_path = os.path.join(sub_dir_path, 'image')
        # print(sub_dir_path)
        if os.path.isdir(sub_dir_path):
            # Iterate over the image files in the current subdirectory
            image_files = []
            for file in os.listdir(sub_dir_path):
                if file.endswith(".png"):
                    image_files.append(os.path.join(sub_dir_path, file))
            
            # Shuffle the image files
            random.shuffle(image_files)
            
            # Split the image files into three parts: train, val, and test
            num_files = len(image_files)
            train_split = int(0.7 * num_files)  # 70% for training
            val_split = int(0.2 * num_files)   # 20% for validation
            train_data_file_names.extend(image_files[:train_split])
            val_data_file_names.extend(image_files[train_split:train_split+val_split])
            test_data_file_names.extend(image_files[train_split+val_split:])

    # Print the resulting lists
    # print("Train data files:")
    # for file_name in train_data_file_names:
    #     print(file_name)

    # print("\nValidation data files:")
    # for file_name in val_data_file_names:
    #     print(file_name)

    # print("\nTest data files:")
    # for file_name in test_data_file_names:
    #     print(file_name)
    return train_data_file_names, val_data_file_names, test_data_file_names


class ESD_Dataset(Dataset):
    def __init__(self, file_names, ids=False):
        self.file_names = file_names
        self.image_transform = image_transform()
        self.mask_transform = mask_transform()
        self.ids = ids
        self.transforms = ToTensor()
        
        


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]

        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        data = {"image": image, "mask": mask}
        augmented = self.mask_transform(**data)
        mask = augmented["mask"]
        image = self.image_transform(image=image)

        image = image['image']
        # print(image.shape)
        image = self.transforms(image)
        label = torch.from_numpy(mask).long()
        # print(image.shape, label.shape)
        sample = {'image': image, 'label': label, 'id': str(img_file_name).split('\\')[-1]}
        if self.ids: 
            return sample['image'], sample['label'], sample['id']
        return sample['image'], sample['label']

    def get_N(self, split=[.6, .8]): 
        class_index, class_count = np.unique(self.Y[:int(split[0] * self.n_data)], return_counts=True)
        if self.output_dim is not None:
            N = np.zeros(self.output_dim)
            N[class_index.astype(int)] = class_count
            N = torch.tensor(N)
        else:
            N = None

def load_image(path):
    # print(str(path))
    img = cv2.imread(str(path))
    # print('Done Done Done')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_folder = 'mask'
    path = str(path).replace('image', mask_folder)
    # print(path)
    identifier = path.split('/')[-1]
    path = path.replace(identifier, identifier[:-4] + '_mask' + '.png')
    # mask = cv2.imread(path, 0)
    mask = cv2.imread(str(path), 0)
    # mask = (mask / factor).astype(np.uint8)
    # print(np.unique(mask))
    # if len(np.unique(mask)) == 7:
    #     print(np.unique(mask))

    # print(mask.all)
    # print(mask == 0)
    # print("____________________")
    # np.set_printoptions(threshold=np.inf)
    # print(mask == 255)
    mask[mask == 255] = 4
    mask[mask == 212] = 0
    mask[mask == 170] = 0
    mask[mask == 128] = 3
    mask[mask == 85] = 2
    mask[mask == 42] = 1
    # print(mask.all)
    
    return mask.astype(np.uint8)

