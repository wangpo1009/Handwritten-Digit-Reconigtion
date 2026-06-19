from torch.utils.data import Dataset
import struct
import numpy as np
import torch
from pathlib import Path

# Data file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "mnist"

def load_images(filepath):
    with open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype = np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

def load_labels(filepath):
    with open(filepath, "rb") as f:
        magic,num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype = np.uint8)
    return labels

def load_mnist():
    train_images = load_images(DATA_RAW_PATH / "train-images.idx3-ubyte")
    train_labels = load_labels(DATA_RAW_PATH/"train-labels.idx1-ubyte")
    test_images = load_images(DATA_RAW_PATH/"t10k-images.idx3-ubyte")
    test_labels = load_labels(DATA_RAW_PATH/"t10k-labels.idx1-ubyte")
    return (train_images, train_labels, test_images, test_labels)

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.astype(np.float32)/255 # Đưa về [0,1]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype = torch.float32).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype = torch.long)
        return image, label
        

