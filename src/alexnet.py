import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from load_sim_and_gen_data import OriginalTrainDataset, TestDataset, GeneratedDataset, CombinedTrainDataset  
from load_sim_and_gen_data import GEOM_CLASSES


class AlexNet1D(nn.Module):
    "A 1D ConvNet based on AlexNet kernel size, stride, and padding."

    def __init__(self, num_classes=4):
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
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, trainset, n_epochs, print_every_n_batches=100, batch_size=64,
        save_model_dir="model/"):
    """Trains and saves a given model."""

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            # zero the parameter gradients, important
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % print_every_n_batches == print_every_n_batches - 1:    # print every 100 mini-batches
                print("[epoch %d, batch %5d] avg loss: %.4f" %
                      (epoch + 1, i + 1, running_loss / (print_every_n_batches *
                          batch_size)))
                running_loss = 0.0

    path = os.path.join(save_model_dir, time.strftime("%Y%m%d-%H%M%S") + ".pth")
    torch.save(model.state_dict(), path)
    print("Model saved to %s." % path)
    return model, path


def evaluate(model_path, testset, class_names, model=None, batch_size=64):
    if model is None:
        model = AlexNet1D()
        model.load_state_dict(torch.load(model_path))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
            shuffle=False, num_workers=1)
    cross_entropy_fn = nn.CrossEntropyLoss()
    cross_entropy_loss = 0.0
    n_classes = len(class_names)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    # no gradients when evalutating
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            cross_entropy_loss += cross_entropy_fn(outputs, labels).item()
            c = (predicted == labels).squeeze()
            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1
    metrics = {}
    for i, c in enumerate(class_names):
        metrics['class_acc_' + c] = float(class_correct[i]) / class_total[i]
    metrics['overall_acc'] = float(np.sum(class_correct)) / np.sum(class_total)
    metrics['avg_cross_entropy_loss'] = cross_entropy_loss / np.sum(class_total)
    metrics_path = model_path.split('.')[0] + '_metrics.csv'
    with open(metrics_path, 'w') as f:
        w = csv.DictWriter(f, metrics.keys())
        w.writeheader()
        w.writerow(metrics)
    print('Metrics on test set, saved to %s.' % metrics_path)
    for k, v in metrics.items():
        print(k, ':', v)


def main():
    og_train_data = OriginalTrainDataset()
    # gen_data = GeneratedDataset()
    # combined_train_data = CombinedTrainDataset()
    test_data = TestDataset()

    model = AlexNet1D()
    model, saved_path = train(model, og_train_data, 100)
    evaluate(saved_path, test_data, GEOM_CLASSES)


if __name__ == "__main__":
    main()
