"""
============================================================

Author: Jason
Description: Utility functions for the Model Hijacking


============================================================

"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torchvision import datasets, transforms
import torchvision
from torchvision.models import mobilenet_v2
from networks import (
    MLP,
    ConvNet,
    LeNet,
    AlexNet,
    VGG11BN,
    VGG16,
    VGG11,
    ResNet18,
    ResNet18BN_AP,
    ResNet18_AP,)
import requests
import zipfile
import pandas as pd


"""

***********************
    Get dataset
***********************

"""


def get_dataset(dataset, data_path):
    if dataset == "mnist":
        channel = 1
        num_classes = 10
        im_size = (28, 28)
        mean = (0.1307,)
        std = (0.3081,)
        transforms = transforms.compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dst_train = datasets.MNIST(
            data_path, train=True, download=True, transform=transforms
        )
        dst_test = datasets.MNIST(
            data_path, train=False, download=True, transform=transforms
        )
        class_names = dst_train.classes

    elif dataset == "cifar10":
        channel = 3
        num_classes = 10
        im_size = (32, 32)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dst_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transforms
        )
        dst_test = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transforms
        )
        class_names = dst_train.classes

    elif dataset == "celeba":
        channel = 3
        num_classes = 40
        im_size = (218, 178)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dst_train = datasets.CelebA(
            data_path, split="train", download=True, transform=transforms
        )
        dst_test = datasets.CelebA(
            data_path, split="valid", download=True, transform=transforms
        )
        class_names = dst_train.attr_names

    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    return dst_train, dst_test, channel, num_classes, im_size, mean, std, class_names


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        lab = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, lab

    def __len__(self):
        return self.images.shape[0]


"""

***********************
    Get networks
***********************

"""


def get_network(name, num_classes, channel, input_size=(32, 32), dist=True):
    """
    @param:
    name(str): the name of the network
    channel(int): image channel
    num_classes(int): the number of classes
    input_size(tuple): the size of the input image
    dist(bool): whether to use distributed training

    @return:
    net(nn.Module): the network instance
    """

    if name == "MLP":
        net = MLP(channel=channel, num_classes=num_classes)
    elif name == "ConvNet":
        net = ConvNet(channel=channel, num_classes=num_classes, input_size=input_size)
    elif name == "LeNet":
        net = LeNet(channel=channel, num_classes=num_classes)
    elif name == "alexnet":
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif name == "VGG11":
        name = VGG11(num_classes=num_classes)
    elif name == "VGG11BN":
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif name == "VGG16":
        net = VGG16(channel=channel, num_classes=num_classes)
    elif name == "ResNet18":
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif name == "ResNet18BN_AP":
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif name == "ResNet18_AP":
        net = ResNet18_AP(channel=channel, num_classes=num_classes)

    elif name == "ConvNetD1":
        net = ConvNet(channel=channel, num_classes=num_classes)
    elif name == "ConvNetD2":
        net = ConvNet(channel=channel, num_classes=num_classes)
    elif name == "ConvNetD3":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetD4":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetD5":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetD6":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetD7":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetD8":
        net = ConvNet(channel=channel, num_classes=num_classes)

    elif name == "ConvNetW32":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetW64":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetW128":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetW256":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetW512":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetW1024":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )

    elif name == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes)

    elif name == "ConvNetAS":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetAR":
        net = ConvNet(channel=channel, num_classes=num_classes)
    elif name == "ConvNetAL":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )

    elif name == "ConvNetNN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetBN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetLN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetIN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetGN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )

    elif name == "ConvNetNP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetMP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )
    elif name == "ConvNetAP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
        )

    else:
        net = None
        exit("Error: unknown model")

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num > 0:
            device = "cuda"
            if gpu_num > 1:
                net = nn.DataParallel(net)
        else:
            device = "cpu"
        net = net.to(device)

    return net


"""

*********************
    Loss function
*********************

"""

def visual_loss(X_c, X_o):
    """
    X_c: tensor, generated by decoder, a camouflaged sample
    X_o: tensor, first encoder take a smple X_o from hijack dataset as its input

    Visual Loss calculates the L1 distance between the output of the Camouflager and the hijackee sample.
    """
    vl = torch.norm(X_c - X_o, p=1)
    return vl


def semantic_loss(X_c, X_h, feature_extractor):
    """
        F is feature extractor, which is a pretrained model (Mark Sandler, Andrew G. Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. MobileNetV2: Inverted Residuals and Linear Bottlenecks).

        Semantic Loss calculates the L1 distance between the features of the output of the Camouflager and the hijacking sample.

    """
    f_xc = feature_extractor(X_c)
    f_xh = feature_extractor(X_h)
    sl = torch.norm(f_xc - f_xh, p=1)

    return sl


def adv_semantic_loss(X_c, X_o, feature_extractor):
    """
    ADV loss maximizes the difference between the features of the hijackee and camouflaged samples using the
    L1 distance.

    """
    f_xc = feature_extractor(X_c)
    f_xo = feature_extractor(X_o)
    asl = torch.norm(f_xc - f_xo, p=1)

    return asl

def camouflage_loss(X_c, X_o,X_h, feature_extractor):
    """


    """
    
    # Compute Euclidean distance between x_c and x_o for each pair
    dist_xc_xo = torch.norm(X_c - X_o, p=2, dim=1)  # Shape: (batch_size,)

    # Apply transformation F to x_c and x_h
    F_xc = feature_extractor(X_c)  # Shape: (batch_size, feature_dim)
    F_xh = feature_extractor(X_h)  # Shape: (num_h, feature_dim)

    # Compute pairwise Euclidean distances between F(x_c) and F(x_h)
    dist_F = torch.cdist(F_xc, F_xh, p=2)  # Shape: (batch_size, num_h)

    # Find the minimum distance over x_h for each x_c
    min_dist_F = torch.min(dist_F, dim=1)[0]  # Shape: (batch_size,)

    # Combine the two terms for each sample
    loss_per_sample = dist_xc_xo + min_dist_F

    # Sum over all samples in the batch
    cham = torch.sum(loss_per_sample)
    return cham


def adv_chameleon(X_c, X_o,X_h, feature_extractor):
    """
    

    """

    F_Xc = feature_extractor(X_c)  
    F_Xh = feature_extractor(X_h)  
    F_Xo = feature_extractor(X_o)  
    dist_Xc_Xo = torch.norm(X_c - X_o, p=2, dim=1)  # Shape: (batch_size,)
    dist_F = torch.cdist(F_Xc, F_Xh, p=2)  # Shape: (batch_size, num_h)
    min_dist_F = torch.min(dist_F, dim=1)[0]  # Shape: (batch_size,)
    dist_F_Xc_Xo = torch.norm(F_Xc - F_Xo, p=2, dim=1)  # Shape: (batch_size,)
    loss_per_sample = dist_Xc_Xo + min_dist_F - dist_F_Xc_Xo
    cham_adv = torch.sum(loss_per_sample)  # Sum over all samples in the batch  
    return cham_adv


"""

**********************
    Evaluation
**********************

"""


"""

************************
    other utils
************************


"""

#! feature extraction (MobileNetV2)
def load_feature_extractor():
    feature_extractor = mobilenet_v2(pretrained=True).features
    feature_extractor = feature_extractor.eval()
    return feature_extractor

#! Camouflager Encoder and Decoder
class Encoder(nn.Module):
    """
    The Camouflager encoders (E_o and E_h) architecture:
    conv2d(k,f) two dimensional convolutional 
    x_in is input sample, kernal_size=4
    activation function: RuLU
    xin ——>Conv2d(4,12)
    Conv2d(4,24)
    Conv2d(4,48)
    Conv2d(4,96)
    ——>μ
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 20, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 48, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 96, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x


class Decoder(nn.Module):
    """
    The Camouflager decoder architecture:
    (μ_o || μ_h) ——>
    ConvTranspose2d(4,96)
    ConvTranspose2d(4,48)
    ConvTranspose2d(4,24)
    ConvTranspose2d(4,3)
    ——> xout
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(96)
        self.deconv2 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(48)
        self.deconv3 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(24)
        self.deconv4 = nn.ConvTranspose2d(24, 3, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x


class Camouflager(nn.Module):
    def __init__(self):
        super(Camouflager, self).__init__()
        self.encoder_o = Encoder()
        self.encoder_h = Encoder()
        self.decoder = Decoder()

    def forward(self, x_o, x_h):
        mu_o = self.encoder_o(x_o)
        mu_h = self.encoder_h(x_h)
        concatenated = torch.cat((mu_o, mu_h), dim=1) 
        x_out = self.decoder(concatenated)
        return x_out


#! get time
def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
