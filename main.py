import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt
from utils import (
    get_dataset,
    get_time,
    get_network,
    visual_loss,
    semantic_loss,
    adv_semantic_loss,
    adv_chameleon,
    fea_extra,
)
from UNet.unet_model import Camouflager


wandb.init(project="model_hijacking")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def save_images(camouflaged, filename="camouflaged_samples.png"):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        camo_img = camouflaged[i].permute(1, 2, 0).cpu().numpy() * np.array(
            [0.2023, 0.1994, 0.2010]
        ) + np.array([0.4914, 0.4822, 0.4465])
        axes[i].imshow(np.clip(camo_img, 0, 1))
        axes[i].set_title("Camouflaged")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Images saved to {filename}")


def main(args):

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    print("Hyper-parameters: \n", args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_camouflager(
        camouflager,
        dataloader_o,
        dataloader_h,
        feature_extractor,
        attack_type=args.attack_type,
        epochs=args.epochs,
        lr=args.learning_rate,
    ):
        device = next(camouflager.parameters()).device
        optimizer = optim.Adam(camouflager.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            for (x_o, _), (x_h, _) in zip(dataloader_o, dataloader_h):
                x_o, x_h = x_o.to(device), x_h.to(device)
                optimizer.zero_grad()
                x_c = camouflager(x_o, x_h)
                vl = visual_loss(x_c, x_o)
                sl = semantic_loss(x_c, x_h, feature_extractor)
                loss = vl + sl
                if attack_type == "adv_chameleon":
                    asl = adv_chameleon(x_c, x_o, feature_extractor)
                    loss += asl
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                #! Visualize samples every 100 epochs
                if epoch % 100 == 0 and epoch > 0:
                    save_images(x_c, filename=f"camouflaged_epoch_{epoch}.png")

            avg_loss = total_loss / len(dataloader_o)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            wandb.log({"Camouflager Loss": avg_loss})

    def generate_camouflaged_samples(camouflager, dataloader_h, dataloader_o):
        camouflaged_samples = []
        with torch.no_grad():
            for (x_h, y_h), (x_o, _) in zip(dataloader_h, dataloader_o):
                x_h, x_o = x_h.to(device), x_o.to(device)
                x_c = camouflager(x_o, x_h)
                camouflaged_samples.append((x_c, y_h))
        return camouflaged_samples

    def train_target_model(model, dataloader, criterion, optimizer, epochs=5):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(
                f"Target Model Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}"
            )
            wandb.log({"Target Model Loss": avg_loss})

    def evaluate_model(model, dataloader, task="original"):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        acc = 100 * correct / total
        print(f"{task} Task Accuracy: {acc:.4f}%")
        return acc

    wandb.config.update(args)
    """

    LOADING

    """
    print("%s LOADING DATASETS" % get_time())

    (
        train_hijackee,
        test_hijackee,
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
    ) = get_dataset(
        args.original_dataset, args.data_path
    )  # hijackee dataset
    train_hijacker, test_hijacker, _, _, _, _, _, _, _, _, _ = get_dataset(
        args.hijacking_dataset, args.data_path
    )  # hijacker dataset

    train_loader_o = torch.utils.data.DataLoader(
        train_hijackee, batch_size=args.batch_size, shuffle=True
    )
    test_loader_o = torch.utils.data.DataLoader(
        test_hijackee, batch_size=args.batch_size, shuffle=False
    )
    train_loader_h = torch.utils.data.DataLoader(
        train_hijacker, batch_size=args.batch_size, shuffle=True
    )
    test_loader_h = torch.utils.data.DataLoader(
        test_hijacker, batch_size=args.batch_size, shuffle=False
    )

    camouflager = Camouflager(n_channels=channel, n_classes=num_classes).to(device)
    feature_extractor = fea_extra().to(device)

    """

    TRAINING

    """

    print("%s TRAINING BEGINS" % get_time())
    train_camouflager(
        camouflager,
        train_loader_o,
        train_loader_h,
        feature_extractor,
        args.attack_type,
        epochs=args.epochs,
        lr=args.learning_rate,
    )

    camouflaged_samples = generate_camouflaged_samples(
        camouflager, train_loader_h, train_loader_o
    )

    poisoned_dataset = []
    for x_c, y_h in camouflaged_samples:
        poisoned_dataset.append((x_c, y_h))
    poisoned_dataset.extend([(x, y) for (x, y) in train_hijackee])
    poisoned_loader = torch.utils.data.DataLoader(
        poisoned_dataset, batch_size=args.batch_size, shuffle=True
    )

    target_model = get_network(args.model, num_classes, channel).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(target_model.parameters(), lr=args.learning_rate)
    train_target_model(
        target_model,
        poisoned_loader,
        criterion,
        optimizer,
        train_epochs=args.train_epochs,
    )

    """
    
    EVALUATION

    """
    print("%s EVALUATION BEGINS" % get_time())
    print("Evaluating on Original Dataset")
    utility = evaluate_model(target_model, test_loader_o, task="original")

    print("Evaluating on Hijacking Dataset")
    camouflaged_test_samples = generate_camouflaged_samples(
        camouflager, test_loader_h, test_loader_o
    )
    camouflaged_test_loader = torch.utils.data.DataLoader(
        camouflaged_test_samples, batch_size=args.batch_size, shuffle=False
    )
    attack_acc = evaluate_model(target_model, camouflaged_test_loader, task="hijacking")

    print("\n ==========================Final Results==========================\n")
    print("Utility: {:.4f}%".format(utility))
    print("Attack Accuracy: {:.4f}%".format(attack_acc))

    wandb.log({"Utility Accuracy": utility})
    wandb.log({"Attack Accuracy": attack_acc})

    wandb.finish()


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
        "--train_epochs",
        type=int,
        default=100,
        help="Number of epochs for training target model",
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
