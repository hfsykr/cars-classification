from torch.utils.data import Dataset
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch

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


    
