import argparse
import csv
from datetime import datetime
from math import ceil
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np

from load_sim_and_gen_data import OriginalTrainDataset, ValidationDataset, TestDataset
from load_sim_and_gen_data import GeneratedDataset, CombinedTrainDataset  

from alexnet import AlexNet1D
from fc import OneLayerFC, TwoLayerFC

MODEL_NAME_TO_CLASS_MAP = {
    'alexnet': AlexNet1D,
    'onelayerfc': OneLayerFC,
    'twolayerfc': TwoLayerFC,
}

N_GEOM_CLASSES = ValidationDataset.n_geom_classes()
N_MAT_CLASSES = ValidationDataset.n_mat_classes()


def loss_fn(logits, labels_geom, labels_mat):
    assert logits.shape[1] == N_GEOM_CLASSES + N_MAT_CLASSES
    ce = nn.CrossEntropyLoss()
    return ce(logits[:, :N_GEOM_CLASSES], labels_geom) + ce(
            logits[:,-N_MAT_CLASSES:], labels_mat)


def train(model, trainset, n_epochs, print_every_n_batches=100, batch_size=64,
        learning_rate=1e-4, save_model_dir="model/", validation_set=None,
        summary_writer=None):
    """Trains and saves a given model."""

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if summary_writer is None:
        summary_writer = SummaryWriter()

    n_batches_per_epoch = int(ceil(float(trainset.__len__()) / batch_size))
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels_geom, labels_mat = data
            # zero the parameter gradients, important
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels_geom, labels_mat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % print_every_n_batches == print_every_n_batches - 1:    # print every 100 mini-batches
                avg_loss = running_loss / (print_every_n_batches * batch_size)
                print("[epoch %d, batch %5d] avg loss: %.6f" %
                      (epoch, i, avg_loss))
                summary_writer.add_scalar('loss/train', avg_loss,
                        global_step=epoch*n_batches_per_epoch+i)
                running_loss = 0.0
        if validation_set is not None:
            metrics = compute_metrics(model, validation_set)
            for k, v in metrics.items():
                summary_writer.add_scalar(k, v, epoch*n_batches_per_epoch)

    path = os.path.join(save_model_dir, time.strftime("%Y%m%d-%H%M%S") + ".pth")
    torch.save(model.state_dict(), path)
    print("Model saved to %s." % path)
    return model, path

def compute_metrics(model, validation_set, print_metrics=False):
    evalloader = torch.utils.data.DataLoader(validation_set,
            batch_size=64, shuffle=False, num_workers=1)
    metrics = {}
    for geom_or_mat in ['geom', 'mat']:
        if geom_or_mat == 'geom':
            class_names = validation_set.get_geom_class_names()
        else:
            class_names = validation_set.get_mat_class_names()
        cross_entropy_fn = nn.CrossEntropyLoss()
        cross_entropy_loss = 0.0
        n_classes = len(class_names)
        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))
        # no gradients when evalutating
        with torch.no_grad():
            for data in evalloader:
                if geom_or_mat == 'geom':
                    inputs, labels, _ = data
                    outputs = model(inputs)[:, :N_GEOM_CLASSES]
                else:
                    inputs, _, labels = data
                    outputs = model(inputs)[:, -N_MAT_CLASSES:]
                predicted = torch.argmax(outputs.data, dim=1)
                cross_entropy_loss += cross_entropy_fn(outputs, labels).item()
                c = (predicted == labels).squeeze()
                for i, label in enumerate(labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i, c in enumerate(class_names):
            metrics['accuracy/' + c] = float(class_correct[i]) / class_total[i]
            metrics['n_examples/' + c] = class_total[i]
        metrics['accuracy/avg_' + geom_or_mat] = float(
                np.sum(class_correct)) / np.sum(class_total)
        metrics['loss/validation_' + geom_or_mat] = (
                cross_entropy_loss / np.sum(class_total))
    metrics['loss/validation'] = (
            metrics['loss/validation_geom'] + metrics['loss/validation_mat'])
    if print_metrics:
        for k, v in metrics.items():
            print(k, ':', v)
    return metrics


def evaluate(model_path, validation_set, model_cls):
    model = model_cls(validation_set.n_logits())
    model.load_state_dict(torch.load(model_path))
    metrics = compute_metrics(model, validation_set, print_metrics=True)
    metrics_path = model_path.split('.')[0] + '_metrics.csv'
    with open(metrics_path, 'w') as f:
        w = csv.DictWriter(f, metrics.keys())
        w.writeheader()
        w.writerow(metrics)
    print('Metrics saved to %s.' % metrics_path)


def main():
    parser = argparse.ArgumentParser(description='Training config.')
    parser.add_argument('--model_name', type=str, default='alexnet',
            help='Model name, available options: %s.' % (
            ', '.join(MODEL_NAME_TO_CLASS_MAP.keys())))
    parser.add_argument('--exclude_gen_data', default=False,
            action='store_true', help='Whether to exclude generated data '
            'and only use original training data.')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Learning rate.')
    parser.add_argument('--n_epochs', type=int, default=20,
            help='Number of epochs in training.')
    args = parser.parse_args()

    if args.exclude_gen_data:
        train_set = OriginalTrainDataset()
    else:
        train_set = CombinedTrainDataset()
    validation_set = ValidationDataset()

    try:
        model_cls = MODEL_NAME_TO_CLASS_MAP[args.model_name.lower()]
    except:
        print('Unrecognized model name.')
        return
    model = model_cls(n_logits=validation_set.n_logits())

    dt = datetime.now().strftime("%m_%d_%Y_%H:%M")
    writer = SummaryWriter(log_dir="runs/%s-lr_%f-epochs_%d-trainsize_%d-%s" %
            (args.model_name, args.lr, args.n_epochs, train_set.__len__(), dt))
    print('Logging training progress to tensorboard dir %s.' % writer.log_dir)
    model, saved_path = train(model, train_set, args.n_epochs,
            validation_set=validation_set, summary_writer=writer)
    evaluate(saved_path, validation_set, model_cls)


if __name__ == "__main__":
    main()
