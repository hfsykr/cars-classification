from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch
import matplotlib.pyplot as plt

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

def get_model(n_class):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_ftrs, n_class)

    return model

def save_figure(output, title, x_label, y_label, line, label):
    plt.figure(figsize=(15, 7.5))

    for i in range(len(line)):
        plt.plot(line[i], label=label[i])
    plt.title(title, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.legend(prop={'size': 16})
    plt.grid()

    plt.savefig(output, bbox_inches='tight')