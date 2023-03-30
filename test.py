import argparse
from pathlib import Path
from scipy.io import loadmat
from torchvision import transforms
from utils import CarsDataset, get_model
import torch
from torch.utils.data import  DataLoader
import time
from tqdm import tqdm

if '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--data', type=str, default='data', help='path of your dataset location')
    argParser.add_argument('--model', type=str, default='mobilenet_v3', help='model that will be used for testing')
    argParser.add_argument('--weights', type=str, help='path of your trained model weights location')
    argParser.add_argument('--image_size', type=int, default=[240, 360], nargs=2, help='image size (h x w)')
    argParser.add_argument('--device', type=str, default='cuda', help='device used for testing, either cuda (gpu) or cpu')

    args = argParser.parse_args()

    data = Path(args.data)
    images_data = data/'cars_test'

    annot_mat = loadmat(data/'cars_test_annos_withlabels.mat')
    class_mat = loadmat(data/'cars_meta.mat')

    images = [p for p in images_data.iterdir() if p.is_file()]

    # Substracting every label with 1, because by default the label start from 1
    labels = [annot['class'][0][0] - 1 for annot in annot_mat['annotations'][0]]
    # Change from int to long
    labels = torch.as_tensor(labels, dtype=torch.long)

    class_names = [class_name[0] for class_name in class_mat['class_names'][0]]
    n_class = len(class_names)

    image_size = (args.image_size[0], args.image_size[1])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = CarsDataset(images, labels, transform)

    test_loader = DataLoader(test_dataset)

    model = get_model(args.model, n_class)
    
    device = torch.device(args.device)
    model.to(device)

    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)

    total = 0
    correct = 0
    model.eval()

    time_start = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            total += inputs.size(0)
            correct += torch.sum(preds == labels.data).item()
    
    time_elapsed = time.time() - time_start
    acc = (correct / total) * 100

    print(f'Test Accuracy: {acc:.2f} %')
    print('Testing complete in', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))