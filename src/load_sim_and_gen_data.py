import glob
import itertools

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, ConcatDataset

GEOM_CLASSES = ["TriangPrismIsosc", "parallelepiped", "sphere", "wire"]
MAT_CLASSES = ["Au", "SiN", "SiO2"]
DIM_LABELS = ["ShortestDim", "MiddleDim", "LongDim", "log Area/Vol"]


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
        self.X_diff = self.X[:, 1:] - self.X[:, :-1]
        self.X = np.concatenate([self.X, self.X_diff], axis=1)
        df_y = read_split_files(label_filepath, is_pandas=True)
        # Check X and y have the same number of rows.
        self.n_samples = self.X.shape[0]
        assert df_y.shape[0] == self.n_samples
        # Extract geometry and material information.
        self.geom = df_y[["Geometry_" + g for g in GEOM_CLASSES]].to_numpy()
        self.mat = df_y[["Material_" + g for g in MAT_CLASSES]].to_numpy()
        self.dims = df_y[DIM_LABELS].to_numpy()
        # Check that only one type has 1 in each row.
        assert np.all(np.sum(self.geom, axis=1) == 1)
        assert np.all(np.sum(self.mat, axis=1) == 1)
        # Convert one-hot encoding to integer encoding.
        self.geom = np.argmax(self.geom, axis=1)
        self.mat = np.argmax(self.mat, axis=1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.geom[idx], self.mat[idx], self.dims[idx]

    @staticmethod
    def n_geom_classes():
        return 4

    @staticmethod
    def n_mat_classes():
        return 3

    @staticmethod
    def n_dim_labels():
        return 4

    @staticmethod
    def n_logits():
        return 11

    def get_geom_class_names(self):
        return GEOM_CLASSES
    
    def get_mat_class_names(self):
        return MAT_CLASSES
    
    def get_dim_label_names(self):
        return DIM_LABELS


class OriginalTrainDataset(DatasetFromFilepath):
    """The original simulated training dataset for training random forest."""

    def __init__(self):
        super(OriginalTrainDataset, self).__init__(
            'data/sim_train_spectrum_all.csv', 'data/sim_train_labels_all.csv')


class ValidationDataset(DatasetFromFilepath):
    """The original simulated test dataset."""

    def __init__(self):
        super(ValidationDataset, self).__init__(
            'data/sim_validation_spectrum_all.csv',
            'data/sim_validation_labels_all.csv')


class TestDataset(DatasetFromFilepath):
    """The original simulated test dataset."""

    def __init__(self):
        super(TestDataset, self).__init__(
            'data/sim_test_spectrum_all.csv', 'data/sim_test_labels_all.csv')


class GeneratedDataset(ConcatDataset):
    """The dataset generated from random forest."""

    def __init__(self):
        in_files = glob.glob('data/gen_spectrum_all_*-of-*.csv')
        label_files = glob.glob('data/gen_labels_all_*-of-*.csv')
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

    print('Loading validation dataset..')
    validation = ValidationDataset()
    print('Validation dataset size:', validation.__len__())

    print('Loading test dataset..')
    test = TestDataset()
    print('Test dataset size:', test.__len__())

    print('Loading generated dataset..')
    gen = GeneratedDataset()
    print('Generated dataset size:', gen.__len__())

    print('Loading combined train dataset..')
    combined = CombinedTrainDataset()
    print('Combined train dataset size:', combined.__len__())
    

if __name__ == '__main__':
    main()
