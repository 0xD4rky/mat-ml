# utils/dataset.py
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class AlloyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.dat')])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.loadtxt(file_path, delimiter='\t')

        # Create image from data
        image = np.zeros((256, 256), dtype=np.int32)
        for row in data:
            i, j = int(row[0]), int(row[1])
            class_value = int(row[5])
            image[i, j] = class_value

        # Convert image to float32 and normalize
        image = image.astype(np.float32) / np.max(image)
        
        if self.transform:
            image = self.transform(image)

        return image, idx

# Example data transform (if needed)
from torchvision import transforms

def get_data_loader(data_dir, batch_size=16, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = AlloyDataset(data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
