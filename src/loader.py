import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
batch_size = 32
learning_rate = 0.001
num_epochs = 25
input_size = 224

data_dir = '/Users/darky/Documents/mat-ml/dataset'

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

augmented_dataset = datasets.ImageFolder(data_dir, transform=data_augmentation)
combined_dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

labels = dataset.targets

plt.hist(labels, bins=len(class_names), edgecolor="black")
plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.show()

class_counts = np.bincount(labels)
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
plt.title("Class Distribution (Pie Chart)")
plt.show()