import torch
import torch.nn as nn
import torch.optim as optim


class AlexNet1D(nn.Module):
    "A 1D ConvNet based on AlexNet kernel size, stride, and padding."

    def __init__(self, n_logits=4):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(4, 12, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16 * 6, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(48, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_logits),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
