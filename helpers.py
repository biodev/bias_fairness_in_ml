import os

from meps_local import MEPSDataset

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np

from torchmetrics.functional.classification import binary_confusion_matrix, equal_opportunity

from meps_model import UnWeightedLitDNN

import lightning as L

import torch

import torch.nn.functional as F

import pandas as pd

def get_balanced_accuracy(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    thresh: float) -> float :

    """Compute balanced accuracy (average of sensitivity and specificity)."""
    
    cm = binary_confusion_matrix(preds, labels, threshold = thresh)

    tn, fp, fn, tp = cm.flatten()

    sensitivity = tp / (tp + fn) 

    specificity = tn / (tn + fp)

    bal_acc = (sensitivity + specificity) / 2.0

    return bal_acc.item()

def load_lightning_model_by_torch(
    model: torch.nn.Module, 
    ckpt_file: str, 
    ckpt_name: str = "model"
) -> torch.nn.Module:
    """
    Load a Pytorch model checkpoint from either Lightning or Pytorch.

    Unfortunately NVFLARE currently saves the models in standard Pytorch
    even though Lightning was used for training so we need to be able to 
    load either form.
    """
    
    saved_model = torch.load(ckpt_file)

    #main difference between Pytorch and Lightning is the naming of the model in the dictionary
    #and whether 'model.' is prepended to the dictionary keys.
    state_dict = {key.replace("model.", ""): value for key, value in saved_model[ckpt_name].items()}
    
    model.load_state_dict(state_dict)

    return model

def evaluate_cutoffs(
    model: torch.nn.Module, 
    site_names: list, 
    cutoffs: list
) -> dict:
    
    valid_data = MEPSDataset(files=site_names)
    valid_loader = DataLoader(valid_data, batch_size=100000, shuffle=False)

    model.eval()
    
    with torch.no_grad():
        valid_preds = F.sigmoid(model(valid_data.features))
        valid_bacc = np.array([get_balanced_accuracy(valid_preds, valid_data.labels, thresh=thresh) for thresh in cutoffs ])

        best_idx = np.argmax(valid_bacc)

    return {'best_cutoff': cutoffs[best_idx], 'best_bacc': valid_bacc[best_idx]}

def get_best_cutoff_validation(
    model: torch.nn.Module, 
    site_name: str, 
    cutoffs: list, 
    global_model: bool = False
) -> dict:
    """Find the best classification threshold based on validation data.

    Using the MEPS validation dataset, evaluate and choose the threshold value that gives the 
    best balanced accuracy.

    """

    best_cutoff = evaluate_cutoffs(model, ['for_fl/' + site_name + '/validation.csv'], cutoffs)

    return best_cutoff

def evaluate_on_test(
    model: torch.nn.Module, 
    site_names: list, 
    cutoff: list
) -> dict:
    
    test_data = MEPSDataset(files=site_names)
    test_loader = DataLoader(test_data, batch_size=100000, shuffle=False)

    model.eval()
    
    with torch.no_grad():
        test_preds = F.sigmoid(model(test_data.features))
        test_bacc = get_balanced_accuracy(test_preds, test_data.labels, thresh=cutoff)
        test_eo = equal_opportunity(test_preds, test_data.labels, torch.tensor(test_data.group), threshold=cutoff).popitem()[1].item()
            
    return {'bacc' : test_bacc, 'eo' : test_eo}

def evaluate_on_test_by_site(
    model: torch.nn.Module, 
    site_name: str, 
    site_cutoff: dict, 
    global_model: bool = False
) -> pd.DataFrame:
    """Evaluate the model on the MEPS test data.

    Using the MEPS test dataset, evaluate the balanced accuracy and equal opportunity metrics.  

    """

    test_stats = {}
    
    cutoff = site_cutoff['best_cutoff']
    test_stats[site_name] = evaluate_on_test(model, ['for_fl/' + site_name + '/test.csv'], cutoff)

    return pd.DataFrame.from_dict(test_stats, orient='index')

def get_fl_results_by_site(
    base_name: str, 
    model: torch.nn.Module, 
    cutoffs: list, 
    site_names: list, 
    use_best: bool = False
) -> pd.DataFrame:
    """Summarizes FL global model for each site.

    Given the path to a NVFLARE run, load the resulting trained pytorch global model,
    determine the best classification cutoff based on the validation data and evaluate
    the test dataset based on both the balanced accuracy and equal opportunity metrics.

    Args:
        base_name: Path to NVFLARE run
        model: Instance of the pytorch model as used for training.
        cutoffs: List of candidate cutoffs for classification
        site_names: List of site names to evaluate
        use_best: Whether to use the 'best' model as learned using FL

    Returns:
        A pd.DataFrame containing the sites in the index and columns for balanced
          accuracy (bacc) and equal opportunity (eo).
    
    """

    if use_best:
        ckpt_file = base_name + '/server/simulate_job/app_server/best_FL_global_model.pt'
    else:
        ckpt_file = base_name + '/server/simulate_job/app_server/FL_global_model.pt'

    print('Found checkpoint file:' + ckpt_file)

    fit_model = load_lightning_model_by_torch(model, ckpt_file)

    test_list = []
    
    for site_name in site_names:
    
        site_cutoff = get_best_cutoff_validation(fit_model, site_name, cutoffs)

        test_results = evaluate_on_test_by_site(fit_model, site_name, site_cutoff)

        test_list.append(test_results)
    
    return pd.concat(test_list)

def get_results_by_site(
    suff: str, 
    model: torch.nn.Module, 
    cutoffs: list, 
    site_names: list
) -> pd.DataFrame:
    """Summarizes site-specific models for each site

    Given a list of sites and the corresponding suffix for the run, 
    load the resulting trained pytorch global model, determine the best 
    classification cutoff based on the validation data and evaluate
    the test dataset based on both the balanced accuracy and equal opportunity metrics.

    Args:
        suff: Suffix of site run (e.g. 'standard' or 'weighted')
        model: Instance of the pytorch model as used for training.
        cutoffs: List of candidate cutoffs for classification
        site_names: List of site names to evaluate

    Returns:
        A pd.DataFrame containing the sites in the index and columns for balanced
          accuracy (bacc) and equal opportunity (eo).
          
    """

    test_list = []
    
    for site_name in site_names:
    
        ckpt_list = os.listdir(site_name + suff + '/lightning_logs/version_0/checkpoints/')
    
        assert len(ckpt_list) == 1
    
        ckpt_file = site_name + suff + '/lightning_logs/version_0/checkpoints/' + ckpt_list[0]

        print('Found checkpoint file:' + ckpt_file)

        fit_model = load_lightning_model_by_torch(model, ckpt_file, "state_dict")
    
        site_cutoff = get_best_cutoff_validation(fit_model, site_name, cutoffs)

        test_results = evaluate_on_test_by_site(fit_model, site_name, site_cutoff)

        test_list.append(test_results)
    
    return pd.concat(test_list)
