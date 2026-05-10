import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver_CNN import Result_Saver_CNN


class ImageDataset(Dataset):

    def __init__(self, data, dataset_name):
        self.data = data
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        instance = self.data[index]

        image = instance['image']
        label = instance['label']

        image = np.array(image, dtype=np.float32)

        if self.dataset_name == 'MNIST':
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

        elif self.dataset_name == 'ORL':
            image = image[:, :, 0] / 255.0
            image = np.expand_dims(image, axis=0)
            label = label - 1

        elif self.dataset_name == 'CIFAR':
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def load_dataset(dataset_name):
    file_path = f'../../data/stage_3_data/{dataset_name}'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def get_dataset_config(dataset_name):

    if dataset_name == 'MNIST':
        return {
            'input_channels': 1,
            'image_size': (28, 28),
            'num_classes': 10
        }

    elif dataset_name == 'ORL':
        return {
            'input_channels': 1,
            'image_size': (112, 92),
            'num_classes': 40
        }

    elif dataset_name == 'CIFAR':
        return {
            'input_channels': 3,
            'image_size': (32, 32),
            'num_classes': 10
        }


def train_model(model, train_loader, criterion, optimizer, device, epochs):

    train_losses = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    return train_losses


def evaluate_model(model, test_loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall, f1


def main():

    dataset_name = 'CIFAR'

    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_dataset(dataset_name)

    train_dataset = ImageDataset(data['train'], dataset_name)
    test_dataset = ImageDataset(data['test'], dataset_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    config = get_dataset_config(dataset_name)

    model = Method_CNN(
        input_channels=config['input_channels'],
        image_size=config['image_size'],
        num_classes=config['num_classes']
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = train_model(model, train_loader, criterion, optimizer, device, epochs)

    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)

    saver = Result_Saver_CNN(result_folder='../../result/stage_3_result')

    saver.save_loss_curve(train_losses, dataset_name)
    saver.save_metrics(dataset_name, accuracy, precision, recall, f1)
    saver.print_metrics(dataset_name, accuracy, precision, recall, f1)


if __name__ == '__main__':
    main()