import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import lightning as L

import pandas as pd

from meps_model import WeightedLitDNN, UnWeightedLitDNN

import argparse

import os

class MEPSDataset(Dataset):
    def __init__(self, files: list):
        all_df = []

        for f in files:
            df = pd.read_csv(f)
            all_df.append(df)

        merged_df = pd.concat(all_df, ignore_index=True)

        #add in RACE_White, not sure if it should be removed from the dataset...

        self.group = merged_df.RACE_White.values

        label = merged_df.pop('high_utilization')
        label_cat = label.astype("category")
        label_cat = label_cat.cat.reorder_categories(['low', 'high'])

        self.labels = torch.tensor(label_cat.cat.codes.values, dtype=torch.float32).unsqueeze(1)#reshape to (x,1)

        weights = merged_df.pop('instance_weights')

        self.weights = torch.tensor(weights.values, dtype = torch.float32).unsqueeze(1) #reshape to (x,1)

        self.features = torch.from_numpy(merged_df.values).float()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        
        features = self.features[idx, :]
        labels = self.labels[idx,:]
        weights = self.weights[idx,:]

        return features, labels, weights

class MEPSDataModule(L.LightningDataModule):

    def __init__(self, batch_size: int, file_dirs: list):
        super().__init__()

        self.file_dirs = file_dirs
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = MEPSDataset(files=list(map(lambda x: x + '/train.csv', self.file_dirs)))
            self.valid_data = MEPSDataset(files=list(map(lambda x: x + '/validation.csv', self.file_dirs)))
        elif stage == 'validate':
            self.valid_data = MEPSDataset(files=list(map(lambda x: x + '/validation.csv', self.file_dirs)))
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, action="append")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=100, nargs="?")
    parser.add_argument("--weights", action=argparse.BooleanOptionalAction)
    return parser.parse_args()

if __name__ == "__main__":

    args = define_parser()

    dataset_path = args.dataset_path

    assert isinstance(dataset_path, list) == True
    
    batch_size = args.batch_size
    local_epochs = args.local_epochs
    use_weights = args.weights

    if (use_weights is True):
        weight_str = 'weighted'
        lit_model = WeightedLitDNN()
    else:
        weight_str = 'standard'
        lit_model = UnWeightedLitDNN()

    basename_list = list(map(lambda x: os.path.basename(x), dataset_path))
    
    log_path = "_".join(basename_list) + '_' + weight_str + '_logs'
    
    meps_dm = MEPSDataModule(batch_size, dataset_path)
    trainer = L.Trainer(max_epochs=local_epochs, devices=1, 
                        accelerator="cpu",
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                        logger=CSVLogger(save_dir=log_path),
                        log_every_n_steps=5,
                        enable_progress_bar=False)

    trainer.fit(lit_model, datamodule=meps_dm)

