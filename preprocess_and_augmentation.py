import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch.nn as nn
import kornia as K
import kornia.geometry.transform as KT


class RoadDataset(Dataset):
    def __init__(self, image_dir, image_to_label, image_transform=None, label_transform=None, filter_list=[]):
        self.image_dir = image_dir
        self.image_to_label = image_to_label
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.images = [file for file in os.listdir(image_dir) if file in filter_list]
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        label_path = self.image_to_label(image_path)
      
        image = read_image(image_path)
        label = read_image(label_path)[[0], :, :].type(torch.int64)

        if self.image_transform is not None: 
            image = self.image_transform(image)
            
        if self.label_transform is not None: 
            label = self.label_transform(label)
        return image, label

    def __len__(self):
        return len(self.images)


class Train_Augmentation(nn.Module):
    def __init__(self, img_size, normalize=True, n_classes=6): 
        super(Train_Augmentation, self).__init__()
        self.jitter = transforms.ColorJitter(brightness=.25, hue=0., saturation=.25)
        self.blur = transforms.GaussianBlur(kernel_size=(1, 5))
        self.noise = K.augmentation.RandomGaussianNoise(mean=0, std=0.05, p=0.1)

        self.perspect_transform = K.augmentation.RandomPerspective(p=0.5, distortion_scale=0.15, return_transform=True)
        self.rotate_transform = K.augmentation.RandomAffine(degrees=(-5., 5.), translate=(0.05, 0.05), return_transform=True)
        self.img_size = img_size
        self.normalize = normalize
        self.n_classes = n_classes

    def forward(self, images, labels):
        images = self.jitter(images)
        images = self.blur(images)
        images = images/255
        images = torch.clamp(self.noise(images), 0., 1.)
        images, transform = self.perspect_transform(images)

        labels = KT.warp_perspective(labels.type(torch.float32), transform, dsize=self.img_size, mode='nearest')
        images, transform = self.rotate_transform(images)
        labels = KT.warp_affine(labels, transform[:, :2, :], dsize=self.img_size, mode="nearest")
        if self.normalize: 
            images = transforms.Normalize(mean=[0.2794, 0.2954, 0.2932], std=[0.2461, 0.2657, 0.2773])(images)

        labels = labels.type(torch.LongTensor)
        zero_tensor = torch.zeros(labels.size(0), self.n_classes, labels.size(2), labels.size(3))
        labels = zero_tensor.scatter(1, labels, 1.) 

        return images, labels


class Test_Augmentation(nn.Module): 
    def __init__(self, img_size, normalize=True, n_classes=6): 
        super(Test_Augmentation, self).__init__()
        self.img_size = img_size
        self.normalize = normalize
        self.n_classes = n_classes

    def forward(self, images, labels):
        images = images/255
        images = K.enhance.adjust_contrast(images, 1.2)
        images = K.enhance.adjust_saturation(images, 1.2)
        if self.normalize:
            images = transforms.Normalize(mean=[0.2794, 0.2954, 0.2932], std=[0.2461, 0.2657, 0.2773])(images)

        if labels is not None: 
            labels = labels.type(torch.LongTensor)
            zero_tensor = torch.zeros(labels.size(0), self.n_classes, labels.size(2), labels.size(3))
            labels = zero_tensor.scatter(1, labels, 1.) 

        return images, labels
