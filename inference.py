import argparse
from pathlib import Path
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image
from utils import get_model
import torch

if '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--data', type=str, default='data', help='path of your data location')
    argParser.add_argument('--model', type=str, default='mobilenet_v3', help='your chosen trained model type')
    argParser.add_argument('--image', type=str, help='path of the image that you want to inferenced')
    argParser.add_argument('--weights', type=str, help='path of your trained model weights location')
    argParser.add_argument('--image_size', type=int, default=[240, 360], nargs=2, help='image size (h x w)')
    argParser.add_argument('--device', type=str, default='cuda', help='device used for inference, either cuda (gpu) or cpu')

    args = argParser.parse_args()

    data = Path(args.data)
    class_mat = loadmat(data/'cars_meta.mat')
    train = loadmat(data/'cars_train_annos.mat')

    class_names = [class_name[0] for class_name in class_mat['class_names'][0]]
    n_class = len(class_names)

    image_size = (args.image_size[0], args.image_size[1])
    print(image_size)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = Image.open(args.image).convert('RGB')
    
    input = transform(image)

    model = get_model(args.model, n_class)
    
    device = torch.device(args.device)
    model.to(device)

    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        input = input.to(device)
        input = input.unsqueeze(0)

        output = model(input)
        _, pred = torch.max(output, 1)

    print('Predicted label      :', pred.item())
    print('Predicted class name :', class_names[pred])