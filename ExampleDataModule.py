import pytorch_lightning as pl
import h5py
import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class ExampleDataModule(pl.LightningDataModule):
    def __init__(self, data_file: str, out_path: str, batch_size):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.out_path = out_path

    def prepare_data(self):
        x = np.zeros(10)
        y = np.zeros(10)

        print("Splitting data...")
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=.6, random_state=12, stratify=y)

        del x, y

        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, train_size=.5, random_state=12, stratify=y_val)

        print("Saving data...")
        f = h5py.File(os.path.join(self.out_path, "data_train.hdf5"), 'w')
        f.create_dataset("data", data=x_train, dtype='f4')
        f.create_dataset("annot", data=y_train, dtype='i8')

        f = h5py.File(os.path.join(self.out_path, "data_val.hdf5"), 'w')
        f.create_dataset("data", data=x_val, dtype='f4')
        f.create_dataset("annot", data=y_val, dtype='i8')

        f = h5py.File(os.path.join(self.out_path, "data_test.hdf5"), 'w')
        f.create_dataset("data", data=x_test, dtype='f4')
        f.create_dataset("annot", data=y_test, dtype='i8')

        with open(os.path.join(self.out_path, 'labels.txt'), 'w') as f:
            for item in self.classes:
                f.write("%s\n" % item)


    def setup(self, stage: str = None):
        is_hard = '_hard' if self.use_hard else '_easy'

        if stage == 'fit':
            train = h5py.File(os.path.join(self.out_path, "data_train.hdf5"), 'r')
            self.train_dataset = TensorDataset(
                torch.as_tensor(train['data'][()]),
                torch.as_tensor(train['annot'][()])

            val = h5py.File(os.path.join(self.out_path, "data_val.hdf5"), 'r')
            self.val_dataset = TensorDataset(
                torch.as_tensor(val['data'][()]),
                torch.as_tensor(val['annot'][()])

        if stage == 'validate':
            val = h5py.File(os.path.join(self.out_path, "data_val.hdf5"), 'r')
            self.val_dataset = TensorDataset(
                torch.as_tensor(val['data'][()]),
                torch.as_tensor(val['annot'][()])

        if stage == 'test' or stage == 'predict':
            test = h5py.File(os.path.join(self.out_path, "data_test.hdf5"), 'r')
            self.test_dataset = TensorDataset(
                torch.as_tensor(test['data'][()]),
                torch.as_tensor(test['annot'][()]),
                torch.as_tensor(test['snr'][()]))

        with open(os.path.join(self.out_path, 'labels.txt'), "r") as f:
            self.classes = [line.rstrip() for line in f]


    def train_dataloader(self):
        kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          **kwargs)

    def val_dataloader(self):
        kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True}
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          **kwargs)
    def test_dataloader(self):
        kwargs = {'num_workers': 8, 'pin_memory': True}
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          **kwargs)
    def predict_dataloader(self):
        kwargs = {'num_workers': 8, 'pin_memory': True}
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          **kwargs)