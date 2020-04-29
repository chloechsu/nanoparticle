import torch
import torch.nn as nn
import torch.optim as optim


class AlexNet1D(nn.Module):
    "A 1D ConvNet based on AlexNet kernel size, stride, and padding."

    def __init__(self, n_logits):
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
        self.features_on_diff = nn.Sequential(
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
        self.avgpool_on_diff = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16 * 6 * 2, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(48, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_logits),
        )

    def forward(self, x):
        # Conv on the original spectrum.
        x1 = torch.unsqueeze(x[:, :400], dim=1)
        x1 = self.features(x1)
        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)

        # Conv on the diff.
        x2 = torch.unsqueeze(x[:, 400:], dim=1)
        x2 = self.features_on_diff(x2)
        x2 = self.avgpool_on_diff(x2)
        x2 = torch.flatten(x2, 1)

        # FC layers.
        x = self.classifier(torch.cat([x1, x2], dim=1))
        return x
