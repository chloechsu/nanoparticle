import torch
import torch.nn as nn
import torch.optim as optim


class OneLayerFC(nn.Module):
    "A fully-connected network."

    def __init__(self, n_logits):
        super(OneLayerFC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(799, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_logits),
        )

    def forward(self, x):
        return self.classifier(x)


class TwoLayerFC(nn.Module):
    "A fully-connected network."

    def __init__(self, n_logits):
        super(TwoLayerFC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(799, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_logits),
        )

    def forward(self, x):
        return self.classifier(x)


class ThreeLayerFC(nn.Module):
    "A fully-connected network."

    def __init__(self, n_logits):
        super(ThreeLayerFC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(799, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_logits),
        )

    def forward(self, x):
        return self.classifier(x)
