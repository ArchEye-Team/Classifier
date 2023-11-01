import numpy as np
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

        stats_dataset = Dataset(self, train=False)
        self.num_classes = stats_dataset.num_classes
        self.class2id = stats_dataset.class2id
        self.id2class = stats_dataset.id2class
        self.class_weights = stats_dataset.class_weights

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = Dataset(self, train=True)

            train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                                  test_size=0.2,
                                                  stratify=dataset.class_ids)

            self.train = Subset(dataset, train_idx)
            self.val = Subset(dataset, val_idx)

        if stage == 'test':
            self.test = Dataset(self, train=False)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)
