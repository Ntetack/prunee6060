import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)   # 3 canaux RGB
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        # pour images 150x150 → après conv/pool ≈ 32*34*34
        self.fc1 = nn.Linear(32 * 34 * 34, 100)
        self.fc2 = nn.Linear(100, 6)  # 6 classes Intel

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # -> 73x73
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # -> 34x34

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x  # IMPORTANT: pas de softmax ici