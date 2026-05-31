import torch.nn as nn
import torch.nn.functional as F

class DroneCNN(nn.Module):
    def __init__(self, nb_classes=13):
        super(DroneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        # DROPOUT: Randomly shuts off 50% of neurons during training
        # This prevents the model from memorizing the training set.
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, nb_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Applied only during training
        return self.fc2(x)