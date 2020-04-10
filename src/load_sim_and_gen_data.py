import glob
import itertools

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, ConcatDataset

GEOM_CLASSES = ["TriangPrismIsosc", "parallelepiped", "sphere", "wire"]


def read_split_files(base_filepath, is_pandas=False):
    if base_filepath.endswith('.csv'):
        base_filepath = base_filepath[:-4]
    total_rows = 0
    data = []
    for f in glob.glob(base_filepath+'*.csv'):
        if is_pandas:
            data.append(pd.read_csv(f, dtype=np.float32))
        else:
            data.append(np.loadtxt(f, delimiter=',', dtype=np.float32))
    if is_pandas:
        data = pd.concat(data, axis=0)
    else:
        data = np.concatenate(data, axis=0)
    print('Parsed %d rows from %s.' % (data.shape[0], base_filepath))
    return data


class DatasetFromFilepath(Dataset):
    """Abstract class for creating datasets from filepath."""

    def __init__(self, input_filepath, label_filepath):
        super(DatasetFromFilepath).__init__()
        self.X = read_split_files(input_filepath)
        df_y = read_split_files(label_filepath, is_pandas=True)
        self.y = df_y[["Geometry_" + g for g in GEOM_CLASSES]].to_numpy()
        # Check that only one geometry type has 1 in each row.
        assert np.all(np.sum(self.y, axis=1) == 1)
        # Convert one-hot encoding to integer encoding.
        self.y = np.argmax(self.y, axis=1)
        # Check X and y have the same number of rows.
        self.n_samples = self.X.shape[0]
        assert self.y.shape[0] == self.n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class OriginalTrainDataset(DatasetFromFilepath):
    """The original simulated training dataset for training random forest."""

    def __init__(self):
        super(OriginalTrainDataset, self).__init__(
            'data/sim_train_emi_spectral.csv', 'data/sim_train_geom_spectral.csv')


class TestDataset(DatasetFromFilepath):
    """The original simulated test dataset."""

    def __init__(self):
        super(TestDataset, self).__init__(
            'data/sim_test_emi_spectral.csv', 'data/sim_test_geom_spectral.csv')


class GeneratedDataset(ConcatDataset):
    """The dataset generated from random forest."""

    def __init__(self):
        in_files = glob.glob('data/gen_emi_spectral_*-of-*.csv')
        label_files = glob.glob('data/gen_geom_spectral_*-of-*.csv')
        in_files.sort()
        label_files.sort()
        assert len(in_files) == len(label_files)

        super(GeneratedDataset, self).__init__(
            [DatasetFromFilepath(i, l) for i, l in zip(in_files, label_files)])


class CombinedTrainDataset(ConcatDataset):
    """Generated dataset combined with original train dataset."""
    
    def __init__(self):
        super(CombinedTrainDataset, self).__init__(
                [OriginalTrainDataset(), GeneratedDataset()])
        

def main():
    print('Loading original train dataset..')
    original_train = OriginalTrainDataset()
    print('Original train dataset size:', original_train.__len__())

    print('Loading original test dataset..')
    original_test = TestDataset()
    print('Original test dataset size:', original_test.__len__())

    print('Loading generated dataset..')
    gen = GeneratedDataset()
    print('Generated dataset size:', gen.__len__())

    print('Loading combined train dataset..')
    combined = CombinedTrainDataset()
    print('Combined train dataset size:', combined.__len__())
    

if __name__ == '__main__':
    main()
