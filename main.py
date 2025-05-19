import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="model_hijacking", entity="my_entity")





def main(args):
    pass
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Hijacking")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output")
    args = parser.parse_args()

    main(args)