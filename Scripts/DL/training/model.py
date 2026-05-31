import torch
import torch.nn as nn
import torch.nn.functional as F

class DroneCNN(nn.Module):
    def __init__(self, nb_classes=13):
        super(DroneCNN, self).__init__()
        # On utilise une architecture stable sans BatchNorm pour éviter les bugs DirectML
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 128x128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, nb_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)