import argparse
from pathlib import Path
from scipy.io import loadmat
from torchvision import transforms
from utils import CarsDataset, get_model, train_val_split
import torch
from torch.utils.data import DataLoader
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle

def fit(model, dataloaders, criterion, optimizer, epochs, device, output):
    time_start = time.time()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    time_history = []

    best_loss = np.inf
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, epochs-1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item() # Convert from tensor to float for easier saving later
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                # scheduler.step()
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    print(f'New best loss: {best_loss:.4f}, saving...')
                    torch.save(model.state_dict(), output/'best_loss.pt')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print(f'New best acc: {best_acc:.4f}, saving...')
                    torch.save(model.state_dict(), output/'best_acc.pt')

                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

        torch.save(model.state_dict(), output/'latest.pt')

        if (epoch + 1) % 5 == 0:
            print('Saving checkpoint...')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, output/'checkpoint.pt')

        epoch_elapsed = time.time() - epoch_start
        time_history.append(epoch_elapsed)
        print('Time:', time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed)), '\n')
    
    time_elapsed = time.time() - time_start
    print('Training complete in', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

    history = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'time': time_history
    }
    
    return history

if '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--data', type=str, default='data', help='path of your dataset location')
    argParser.add_argument('--output', type=str, default='output', help='path of your output location')
    argParser.add_argument('--model', type=str, default='mobilenet_v3', help='model that will be used for training')
    argParser.add_argument('--epoch', type=int, default=50, help='how many epoch your model will be trained')
    argParser.add_argument('--image_size', type=int, default=[240, 360], nargs=2, help='image size (h x w)')
    argParser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    argParser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for training')
    argParser.add_argument('--momentum', type=float, default=9e-1, help='momentum used for training')
    argParser.add_argument('--checkpoint', type=str, default=None, help='path of your checkpoint file')
    argParser.add_argument('--device', type=str, default='cuda', help='device used for training, either cuda (gpu) or cpu')

    args = argParser.parse_args()

    data = Path(args.data)
    images_data = data/'cars_train'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(args.output)/('train_' + timestamp)
    # Create the directory if not exist
    output.mkdir(parents=True, exist_ok=True)
    
    annot_mat = loadmat(data/'cars_train_annos.mat')
    class_mat = loadmat(data/'cars_meta.mat')

    images = [p for p in images_data.iterdir() if p.is_file()]

    # Substracting every label with 1, because by default the label start from 1
    labels = [annot['class'][0][0] - 1 for annot in annot_mat['annotations'][0]]
    # Change from int to long
    labels = torch.as_tensor(labels, dtype=torch.long)

    class_names = [class_name[0] for class_name in class_mat['class_names'][0]]
    n_class = len(class_names)

    train_images, val_images, train_labels, val_labels = train_val_split(images, labels, val_size=0.2, shuffle=True, random_seed=42)

    image_size = (args.image_size[0], args.image_size[1])
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CarsDataset(train_images, train_labels, train_transform)
    val_dataset = CarsDataset(val_images, val_labels, val_transform)

    batch_size = args.batch_size

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    }

    model = get_model(args.model, n_class)
    
    device = torch.device(args.device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    learning_rate = args.learning_rate
    momentum = args.momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    epochs = args.epoch

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    history = fit(model, dataloaders, criterion, optimizer, epochs, device, output)

    with open(output/'history.pkl', 'wb') as handle:
        pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)