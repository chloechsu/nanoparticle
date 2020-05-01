import torch
import torch.nn as nn
import torch.optim as optim


class AlexNet1D(nn.Module):
    "A 1D ConvNet based on AlexNet kernel size, stride, and padding."

    def __init__(self, n_logits):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(16, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_logits),
        )

    def forward(self, x):
        # Conv on the original spectrum.
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
