import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn.functional as F

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import WarmUpLR
import warnings
warnings.filterwarnings("ignore")
from timm import create_model


EPOCHS = 50
milestones = [20, 40]
lr_gamma = 0.2
def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='VGG19_sgd_cifar10')
    parser.add_argument("--model", type=str, default='vgg19')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--b', type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='training batch size')
    return parser
def train_model(model, trainloader, valloader, criterion, optimizer, scheduler, name, epochs=100):
    global best_val_acc
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if epoch <= args.warm:
                warmup_scheduler.step()

        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, Current LR: {current_lr}')

        val_acc = validate_model(model, valloader)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/' + name + '_best.pth')

        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        plot_metrics(name)

    print('Training done!')


def plot_metrics(name):
    plt.figure(figsize=(12, 5))

    # plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy per Epoch')

    # plt.show()
    plt.savefig('logs/' + name + '_training.png')
    plt.close()


def validate_model(model, valloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFARViT(nn.Module):
    def __init__(self, model='vit_b_16'):
        super(CIFARViT, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        if model == 'vit_b_16':
            self.vit = torchvision.models.vit_b_16(pretrained=True)
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, 10)
        elif model == 'deit_tiny_patch16_224':
            self.vit = create_model('deit_tiny_patch16_224', pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, 10)
        elif model == 'vit_tiny_patch16_224':
            self.vit = create_model('vit_tiny_patch16_224', pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return self.vit(x)


if __name__ == '__main__':
    parser = get_arguments()
    print(parser)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # test set
    testset = torchvision.datasets.CIFAR10(root='../../data/cifar10/', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, pin_memory=True)

    # train / val sets
    trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10/', train=True, download=True, transform=None)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, pin_memory=True)

    if args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif args.model == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif args.model == 'vit_b_16':
        model = CIFARViT(model='vit_b_16')
    elif args.model == 'deit_tiny_patch16_224':
        model = CIFARViT(model='deit_tiny_patch16_224')
    elif args.model == 'vit_tiny_patch16_224':
        model = CIFARViT(model='vit_tiny_patch16_224')
    elif args.model == 'mlp':
        model = MLP()
    elif args.model == 'cnn':
        model = SimpleCNN()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_gamma)
    iter_per_epoch = len(train_dataset) / args.b
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # for plotting
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    train_model(model, trainloader, valloader, criterion, optimizer, scheduler, args.name, epochs=EPOCHS)

    # load and test the best model
    best_model = model
    best_model.load_state_dict(torch.load('models/' + args.name + '_best.pth'))
    test_model(best_model, testloader)
