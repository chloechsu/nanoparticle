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
from resnet import *
from fc import OneLayerFC, TwoLayerFC, ThreeLayerFC

MODEL_NAME_TO_CLASS_MAP = {
    'alexnet': AlexNet1D,
    'onelayerfc': OneLayerFC,
    'twolayerfc': TwoLayerFC,
    'threelayerfc': ThreeLayerFC,
}
MODEL_NAME_TO_CLASS_MAP.update(RESNET_NAME_TO_MODEL_MAP)

N_GEOM_CLASSES = ValidationDataset.n_geom_classes()
N_MAT_CLASSES = ValidationDataset.n_mat_classes()
N_DIM_LABELS = ValidationDataset.n_dim_labels()


def loss_fn(logits, labels_geom, labels_mat, labels_dim, loss_weights=None):
    if loss_weights is None:
        loss_weights = np.ones(3)
    loss_weights = torch.tensor(loss_weights, requires_grad=False)
    assert logits.shape[1] == N_GEOM_CLASSES + N_MAT_CLASSES + N_DIM_LABELS
    losses = torch.zeros(3)
    ce = nn.CrossEntropyLoss()
    losses[0] = ce(logits[:, :N_GEOM_CLASSES], labels_geom)
    losses[1] = ce(logits[:, N_GEOM_CLASSES:N_GEOM_CLASSES+N_MAT_CLASSES],
            labels_mat)
    losses[2] = nn.MSELoss()(logits[:, -N_DIM_LABELS:], labels_dim)
    return torch.sum(losses * loss_weights)


def train(model, trainset, n_epochs, save_model_path, print_every_n_batches=100,
        batch_size=64, optimizer_name='Adam', learning_rate=1e-4,
        validation_set=None, summary_writer=None, loss_weights=None,
        global_step=0):
    """Trains and saves a given model."""

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=1)
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if summary_writer is None:
        summary_writer = SummaryWriter()

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels_geom, labels_mat, labels_dim = data
            # zero the parameter gradients, important
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels_geom, labels_mat, labels_dim,
                    loss_weights=loss_weights)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % print_every_n_batches == print_every_n_batches - 1:    # print every 100 mini-batches
                avg_loss = running_loss / print_every_n_batches
                print("[epoch %d, batch %5d] avg loss: %.6f" %
                      (epoch, i, avg_loss))
                global_step += print_every_n_batches
                summary_writer.add_scalar('loss/train', avg_loss, global_step)
                running_loss = 0.0
        if validation_set is not None:
            metrics = compute_metrics(model, validation_set)
            for k, v in metrics.items():
                summary_writer.add_scalar(k, v, global_step)

    torch.save(model.state_dict(), save_model_path)
    print("Model saved to %s." % save_model_path)
    return model, global_step

def compute_metrics(model, validation_set, print_metrics=False):
    evalloader = torch.utils.data.DataLoader(validation_set,
            batch_size=64, shuffle=False, num_workers=1)
    metrics = {}
    # Evaluate classification performance.
    for geom_or_mat in ['geom', 'mat']:
        if geom_or_mat == 'geom':
            class_names = validation_set.get_geom_class_names()
        else:
            class_names = validation_set.get_mat_class_names()
        cross_entropy_fn = nn.CrossEntropyLoss(reduction='sum')
        cross_entropy_loss = 0.0
        n_classes = len(class_names)
        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))
        # no gradients when evalutating
        with torch.no_grad():
            for data in evalloader:
                if geom_or_mat == 'geom':
                    inputs, labels, _, _ = data
                    outputs = model(inputs)[:, :N_GEOM_CLASSES]
                else:
                    inputs, _, labels, _ = data
                    outputs = model(inputs)[:, N_GEOM_CLASSES:N_GEOM_CLASSES+N_MAT_CLASSES]
                predicted = torch.argmax(outputs.data, dim=1)
                cross_entropy_loss += cross_entropy_fn(outputs, labels).item()
                c = (predicted == labels).squeeze()
                for i, label in enumerate(labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i, c in enumerate(class_names):
            if class_total[i] == 0:
                metrics['accuracy/' + c] = 0.
            else:
                metrics['accuracy/' + c] = float(class_correct[i]) / class_total[i]
            metrics['n_examples/' + c] = class_total[i]
        metrics['accuracy/avg_' + geom_or_mat] = float(
                np.sum(class_correct)) / validation_set.__len__()
        metrics['loss/validation_' + geom_or_mat] = (
                cross_entropy_loss / validation_set.__len__())
    # Evaluate regression performance of dimensions.
    label_names = validation_set.get_dim_label_names()
    mse_fn = nn.MSELoss(reduction='none')
    mae_fn = nn.L1Loss(reduction='none')
    mse_loss = np.zeros(len(label_names))
    mae_loss = np.zeros(len(label_names))
    with torch.no_grad():
        for data in evalloader:
            inputs, _, _, labels = data 
            outputs = model(inputs)[:, -N_DIM_LABELS:]
            mse_loss += np.sum(mse_fn(outputs, labels).numpy(), axis=0)
            mae_loss += np.sum(mae_fn(outputs, labels).numpy(), axis=0)
    mse_loss /= validation_set.__len__()
    mae_loss /= validation_set.__len__()
    for i, c in enumerate(label_names):
        metrics['MSE/' + c] = mse_loss[i]
        metrics['MAE/' + c] = mae_loss[i]
    metrics['loss/validation_dim'] = np.sum(mse_loss) 
    metrics['loss/validation'] = (metrics['loss/validation_geom'] +
            metrics['loss/validation_mat'] + metrics['loss/validation_dim'])
    if print_metrics:
        for k, v in metrics.items():
            print(k, ':', v)
    return metrics


def evaluate(model_path, validation_set, model_cls):
    model = model_cls(n_logits=validation_set.n_logits())
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
    parser.add_argument('--optimizer_name', type=str, default='Adam',
            help='Optimizer name, Adam or SGD.')
    parser.add_argument('--material', type=str, default='Au',
            help='Material to train and test on: `all`, `Au`, `SiN` or `SiO2`.')
    parser.add_argument('--n_epochs', type=int, default=20,
            help='Number of epochs in training.')
    parser.add_argument('--joint_obj', default=False, action='store_true',
            help='Whether to jointly train with shape and dimension predictions.')
    parser.add_argument('--multistage', default=False, action='store_true',
            help='Whether to train in multiple stages.')
    parser.add_argument('--n_epochs_mat', type=int, default=5,
            help='Number of epochs in training for materials.')
    parser.add_argument('--n_epochs_dim', type=int, default=20,
            help='Number of epochs in training for dimensions.')
    args = parser.parse_args()

    if args.exclude_gen_data:
        train_set = OriginalTrainDataset(args.material)
    else:
        train_set = CombinedTrainDataset(args.material)
    validation_set = ValidationDataset(args.material)

    try:
        model_cls = MODEL_NAME_TO_CLASS_MAP[args.model_name.lower()]
    except:
        print('Unrecognized model name.')
        return
    model = model_cls(n_logits=validation_set.n_logits())

    dt = datetime.now().strftime("%m_%d_%Y_%H:%M")
    model_str = "%s-%s-%s-lr_%f-trainsize_%d-%s" % (args.model_name,
            args.material, args.optimizer_name, args.lr, train_set.__len__(), dt)
    if args.multistage:
        log_dir_name += '-multistage'
    if args.joint_obj:
        log_dir_name += '-joint'
    writer = SummaryWriter(log_dir=os.path.join('runs', model_str))
    print('Logging training progress to tensorboard dir %s.' % writer.log_dir)
    saved_path = os.path.join('model', model_str + '.pth')

    global_step = 0
    
    if args.multistage:
        if args.material == 'all':
            # If mixing materials in training data, frist train materials classification.
            model, global_step = train(model, train_set, args.n_epochs_mat,
                    saved_path, learning_rate=args.lr,
                    optimizer_name=args.optimizer_name,
                    validation_set=validation_set, summary_writer=writer,
                    loss_weights=[0., 1., 0.], global_step=global_step)
            evaluate(saved_path, validation_set, model_cls)
        # Train dimension regression.
        model, global_step = train(model, train_set, args.n_epochs_dim,
                saved_path, learning_rate=args.lr,
                optimizer_name=args.optimizer_name,
                validation_set=validation_set, summary_writer=writer,
                loss_weights=[0., 1., 0.01], global_step=global_step)
        evaluate(saved_path, validation_set, model_cls)
    
    if args.joint_obj:
        loss_weights = [1., 1., 0.01]
    else:
        loss_weights = [1., 0., 0.]
    model, _ = train(model, train_set, args.n_epochs, saved_path,
            learning_rate=args.lr, optimizer_name=args.optimizer_name,
            validation_set=validation_set, summary_writer=writer,
            loss_weights=[1., 0., 0.], global_step=global_step)
    evaluate(saved_path, validation_set, model_cls)


if __name__ == "__main__":
    main()
