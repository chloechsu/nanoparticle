import argparse
import csv
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

shapes = ['TriangPrismIsosc', 'parallelepiped', 'sphere', 'wire']


def main():
    trainsizes = []
    avg_acc = []
    for f in glob.glob('model/resnet18-all-Adam-lr_0.0001*_test_metrics.csv'):
        if 'joint' in f or 'nofeature' in f:
            continue
        print(f)
        trainsize = f.split('-')[4]
        assert trainsize.startswith('trainsize')
        if int(trainsize[10:]) in trainsizes:
            print(trainsize[10:])
        trainsizes.append(int(trainsize[10:]))
        df = pd.read_csv(f)
        avg_acc.append(np.mean([df.iloc[0]['accuracy/' + s] for s in shapes]))
    aug_ratio = [int((t - 7950.) / 7950.) for t in trainsizes]
    print(aug_ratio)
    hues = [str(t == 19) for t in aug_ratio]
    plt.figure(figsize=(8, 5))
    ax = sns.scatterplot(x=aug_ratio[::-1], y=avg_acc[::-1], marker='+',
            hue=hues[::-1], s=80)
    ax.legend_.remove()
    plt.xlabel('Data Augmentation Ratio', fontsize=15)
    plt.ylabel('ResNet18-1D Top-1 Accuracy', fontsize=15)
    plt.savefig('plots/resnet18_datasize_plot.png') 


if __name__ == "__main__":
    main()
