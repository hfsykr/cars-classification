from torch.utils.data import Dataset
from PIL import Image
from torchvision import models
import torch
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

class CarsDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        image = self.transform(image)

        return image, label

def get_model(model_type, n_class):
    model = None

    if model_type == 'mobilenet_v3_l':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT )
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(num_ftrs, n_class)
    elif model_type == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, n_class)

    return model

def save_plot(output, title, x_label, y_label, line, label):
    plt.figure(figsize=(15, 7.5))

    for i in range(len(line)):
        plt.plot(line[i], label=label[i])
    plt.title(title, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.legend(prop={'size': 16})
    plt.grid()

    plt.savefig(output, bbox_inches='tight')

def _indexing(x, indices):
    # Indexing for numpy array
    if hasattr(x, 'shape'):
        return x[indices]

    # Indexing for list
    return [x[idx] for idx in indices]

def train_val_split(*arrays, val_size=0.25, shuffle=True, random_seed=1):
    length = len(arrays[0])

    n_val = int(np.ceil(length * val_size))
    n_train = length - n_val

    if shuffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]
    else:
        train_indices = np.arange(n_train)
        val_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, val_indices)) for x in arrays))