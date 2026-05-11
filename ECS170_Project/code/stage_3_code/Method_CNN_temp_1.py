import torch
import torch.nn as nn
import torch.nn.functional as F


class Method_CNN(nn.Module):
    """
    Deeper CNN model based on the LeNet-style architecture.
    This version adds one additional convolutional layer.
    """

    def __init__(self, input_channels, image_size, num_classes):
        super(Method_CNN, self).__init__()

        # Conv Layer 1
        # input channels = input_channels
        # output channels = 6
        # kernel size = 5x5
        # padding = 2
        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)

        # Max Pooling Layer
        # kernel size = 2x2
        # stride = 2
        self.pool = nn.MaxPool2d(2, 2)

        # Conv Layer 2
        # input channels = 6
        # output channels = 16
        # kernel size = 5x5
        # padding = 2
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)

        # New Conv Layer 3
        # input channels = 16
        # output channels = 32
        # kernel size = 5x5
        # padding = 2
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)

        height, width = image_size

        # after conv1: kernel=5, padding=2, size stays the same

        # after pool1: size is divided by 2
        height = height // 2
        width = width // 2

        # after conv2: kernel=5, padding=2, size stays the same

        # after pool2: size is divided by 2
        height = height // 2
        width = width // 2

        # after conv3: kernel=5, padding=2, size stays the same
        flatten_dim = 32 * height * width

        # Fully Connected Layers
        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))

        # Conv3 -> ReLU
        x = F.relu(self.conv3(x))

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)

        return x
