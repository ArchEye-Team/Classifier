from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from .dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers, image_size):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

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
            self.train, self.val = random_split(dataset, lengths=[0.8, 0.2])
        if stage == 'test':
            self.test = Dataset(self, train=False)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
