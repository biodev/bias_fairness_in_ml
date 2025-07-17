import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import lightning as L

from torchmetrics.functional.classification import binary_confusion_matrix, equal_opportunity

import pandas as pd

import numpy as np

import nvflare.client.lightning as flare

from meps_model import UnWeightedLitDNN, WeightedLitDNN

from meps_local import MEPSDataModule

import argparse

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, nargs="?")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=100, nargs="?")
    parser.add_argument("--weights", action=argparse.BooleanOptionalAction)
    return parser.parse_args()

if __name__ == "__main__":

    args = define_parser()

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    local_epochs = args.local_epochs

    use_weights = args.weights

    if (use_weights is True):
        model = WeightedLitDNN()
    else:
        model = UnWeightedLitDNN()
    
    meps_dm = MEPSDataModule(batch_size, [dataset_path])
    trainer = L.Trainer(max_epochs=local_epochs, devices=1, 
                        accelerator="cpu",
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                        log_every_n_steps=5)
    
    
    # (2) patch the lightning trainer
    flare.patch(trainer)

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        # Note that we don't need to pass this input_model to trainer
        # because after flare.patch the trainer.fit/validate will get the
        # global model internally
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")

        # (4) evaluate the current global model to allow server-side model selection
        print("--- validate global model ---")
        trainer.validate(model, datamodule=meps_dm)

        # perform local training starting with the received global model
        print("--- train new model ---")
        trainer.fit(model, datamodule=meps_dm)


