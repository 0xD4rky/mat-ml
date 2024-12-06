import os
import numpy
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision import datasets,models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import Adam
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


def get_data_loaders(data_dir, input_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    augmented_dataset = datasets.ImageFolder(data_dir, transform=data_augmentation)
    combined_dataset = ConcatDataset([dataset, augmented_dataset])

    train_size = int(0.7 * len(combined_dataset))
    val_size = int(0.15 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.class_to_idx

def plot_class_distribution(dataset):
    labels = []
    for sub_dataset in dataset.datasets:
        if hasattr(sub_dataset, "targets"):
            labels.extend(sub_dataset.targets)
    labels = [int(label) for label in labels]
    class_names = list(dataset.datasets[0].class_to_idx.keys())
    plt.hist(labels, bins=len(class_names), edgecolor="black")
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.show()
