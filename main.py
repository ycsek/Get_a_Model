import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import warnings
from utils import get_dataset, get_time, visual_loss, semantic_loss, adv_semantic_loss,fea_extra


wandb.init(project="model_hijacking")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    print("Hyper-parameters: \n", args.__dict__)
    
    wandb.config.update(args)
    """

    LOAD DATASET

    """
    print("%s LOADING DATASETS" % get_time())

    train_hijackee = (channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader) = get_dataset(
        args.original_dataset, args.data_path
    )  # hijackee dataset
    train_hijacker = (channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader) = get_dataset(
        args.hijacking_dataset, args.data_path
    )  # hijacker dataset

    print("%s TRAINING BEGINS" % get_time())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Hijacking")
    parser.add_argument(
        "--original_dataset",
        type=str,
        default="cifar10",
        help="Dataset (cifar10, mnist, celeba)",
    )
    parser.add_argument(
        "--hijacking_dataset",
        type=str,
        default="mnist",
        help="Dataset (cifar10, mnist, celeba)",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="chameleon",
        help="Attack Type(chameleon, adv_chameleon)",
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", help="Model architecture"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )

    parser.add_argument(
        "--data_path", type=str, default="./data/", help="Path to the dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training (cuda:0, cuda:1, cpu)",
    )
    args = parser.parse_args()

    main(args)
