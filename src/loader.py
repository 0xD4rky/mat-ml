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
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


config = {
    "device" : torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    "batch_size" : 32,
    "learning_rate" : 1e-3,
    "num_epochs" : 25,
    "input_size" : 224,
    "data_dir" : "/Users/darky/Documents/mat-ml/dataset"
}
def loader(data_dir):
    """
    creating dataloaders!
    """
    transform = transforms.Compose([
        transforms.resize((config["input_size"],config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(config[data_dir], transform = transform)
    class_names = datasets.classes

    train_size = int(0.7*(len(dataset)))
    val_size = int(0.15*len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size = config["batch_size"], shuffle = True)
    val_loader = DataLoader(val_set, batch_size = config["batch_size"], shuffle = False)
    test_loader = DataLoader(test_set, batch_size = config["batch_size"], shuffle = False)
    print(f"DataLoaders created!")

loader(config["data_dir"])



