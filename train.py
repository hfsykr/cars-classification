import argparse
from pathlib import Path
from scipy.io import loadmat
from torchvision import transforms
from utils import CarsDataset, get_model, save_figure
import torch
from torch.utils.data import random_split, DataLoader
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle

def fit(model, dataloaders, criterion, optimizer, scheduler, epochs, device, output):
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
        print('Epoch {}/{}'.format(epoch, epochs-1))
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
                running_corrects += torch.sum(preds == labels.data).item() # Convert from tensor to float for easier saving later
            
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
                    print(f'New best loss: {best_loss:.4f}, saving...')
                    torch.save(model.state_dict(), output/'best_loss.pt')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    print(f'New best acc: {best_acc:.4f}, saving...')
                    torch.save(model.state_dict(), output/'best_acc.pt')

                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

        torch.save(model.state_dict(), output/'latest.pt')

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
    argParser.add_argument('--size', type=int, default=[224, 224], nargs=2, help='image input size')
    argParser.add_argument('--epoch', type=int, default=50, help='how many epoch your model will be trained')
    argParser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    argParser.add_argument('--learning_rate', type=float, default=1e-3, help='batch size for training')
    argParser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay used for training')

    args = argParser.parse_args()

    data = Path(args.data)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(args.output)/('train_' + timestamp)
    # Create the directory if not exist
    output.mkdir(parents=True, exist_ok=True)
    
    image_path = data/'cars_train'
    
    annotation_mat = loadmat(data/'cars_train_annos.mat')
    class_mat = loadmat(data/'cars_meta.mat')

    images = [p for p in image_path.iterdir() if p.is_file()]
    image_size = tuple(args.size)

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

    split_generator = torch.Generator().manual_seed(2023)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=split_generator)

    batch_size = args.batch_size

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    }

    model = get_model(n_class)
    
    device = torch.device('cuda:0')

    criterion = torch.nn.CrossEntropyLoss()
    
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = args.epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs,  steps_per_epoch=len(dataloaders['train']))
    
    history = fit(model, dataloaders, criterion, optimizer, scheduler, epochs, device, output)

    # Saving the figure (maybe will be used in the future)
    save_figure(
        output=output/'plot_loss.png', 
        title='Training Loss', 
        x_label='Epoch', 
        y_label='Loss', 
        line=[history['train_loss'], history['val_loss']], 
        label=['Train Loss', 'Val Loss']
    )
    save_figure(
        output=output/'plot_accuracy.png', 
        title='Training Accuracy', 
        x_label='Epoch', 
        y_label='Accuracy', 
        line=[history['train_acc'], history['val_acc']], 
        label=['Train Accuracy', 'Val Accuracy']
    )

    with open(output/'history.pkl', 'wb') as handle:
        pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)