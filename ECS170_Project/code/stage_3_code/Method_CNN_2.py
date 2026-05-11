import torch
import torch.nn as nn


class Method_CNN(nn.Module):
    """
    Baseline CNN model for image classification.

    This model can be used for MNIST, ORL, and CIFAR by changing:
    - input_channels
    - image_size
    - num_classes
    """

    def __init__(self, input_channels, image_size, num_classes,
                 conv1_channels=32, conv2_channels=64, conv3_channels= 128, hidden_dim=128):
        super(Method_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=conv1_channels,
                               kernel_size=3,
                               padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2)

        self.conv2 = nn.Conv2d(in_channels=conv1_channels,
                               out_channels=conv2_channels,
                               kernel_size=3,
                               padding=1)

        self.conv3 = nn.Conv2d(in_channels=conv2_channels,
                               out_channels=conv3_channels,
                               kernel_size=3,
                               padding=1)

        self.relu = nn.ReLU()

        height, width = image_size
        height_after_pool = height // 8
        width_after_pool = width // 8
        flatten_dim = conv3_channels * height_after_pool * width_after_pool

        self.fc1 = nn.Linear(flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x