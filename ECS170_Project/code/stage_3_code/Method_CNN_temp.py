import torch
import torch.nn as nn
import torch.nn.functional as F


class Method_CNN(nn.Module):
    """
    CNN model based on the classic LeNet-style architecture.
    """

    def __init__(self, input_channels, image_size, num_classes):
        super(Method_CNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        height, width = image_size

        # calculate output size after conv/pool operations
        height = ((height - 4) // 2)
        height = ((height - 4) // 2)

        width = ((width - 4) // 2)
        width = ((width - 4) // 2)

        flatten_dim = 16 * height * width

        self.fc1 = nn.Linear(flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
