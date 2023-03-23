from pathlib import Path
from scipy.io import loadmat
from torchvision import transforms
from utils import CarsDataset, get_model
import torch
from torch.utils.data import random_split, DataLoader
import time
from datetime import datetime
from tqdm import tqdm

def fit(model, dataloaders, criterion, optimizer, scheduler, epochs, device, output_path):
    time_start = time.time()

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

            print(f'{phase.capitalize()}. Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                scheduler.step()

        epoch_elapsed = time.time() - epoch_start
        print('Time:', time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed)), '\n')
    
    time_elapsed = time.time() - time_start
    print('Training complete in', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    
    return model

if '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_path = Path('data')
    output_path = Path('output/train')/timestamp
    image_path = data_path/'cars_train'
    annotation_mat = loadmat(data_path/'cars_train_annos.mat')
    class_mat = loadmat(data_path/'cars_meta.mat')

    images = [p for p in image_path.iterdir() if p.is_file()]

    # Substracting every label with 1, because by default the label start from 1
    labels = [annot['class'][0][0] - 1 for annot in annotation_mat['annotations'][0]]
    # Change from int to long
    labels = torch.as_tensor(labels, dtype=torch.long)

    class_names = [class_name[0] for class_name in class_mat['class_names'][0]]
    n_class = len(class_names)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = CarsDataset(images, labels, transform)

    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=split_generator)

    batch_size = 8

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