from pathlib import Path
from scipy.io import loadmat
from torchvision import transforms
from utils import CarsDataset, get_model
import torch
from torch.utils.data import random_split, DataLoader
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

def fit(model, dataloaders, criterion, optimizer, scheduler, epochs, device, output_path):
    time_start = time.time()

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    time_history = []

    best_loss = np.inf
    best_acc = 0.0

    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for _, data in enumerate(tqdm(dataloaders[phase])):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), output_path/'best_loss.pt')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), output_path/'best_acc.pt')

                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

        torch.save(model.state_dict(), output_path/'latest.pt')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, output_path/'checkpoint.pt')

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
    data_path = Path('data')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path('output/train_' + timestamp)
    # Create the directory if not exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_path = data_path/'cars_train'
    
    annotation_mat = loadmat(data_path/'cars_train_annos.mat')
    class_mat = loadmat(data_path/'cars_meta.mat')

    images = [p for p in image_path.iterdir() if p.is_file()]
    image_size = (112, 112)

    # Substracting every label with 1, because by default the label start from 1
    labels = [annot['class'][0][0] - 1 for annot in annotation_mat['annotations'][0]]
    # Change from int to long
    labels = torch.as_tensor(labels, dtype=torch.long)

    class_names = [class_name[0] for class_name in class_mat['class_names'][0]]
    n_class = len(class_names)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = CarsDataset(images, labels, transform)

    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=split_generator)

    batch_size = 32

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    }

    model = get_model(n_class)
    
    device = torch.device('cuda:0')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    epochs = 25
    
    history = fit(model, dataloaders, criterion, optimizer, scheduler, epochs, device, output_path)