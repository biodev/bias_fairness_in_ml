# Evaluation of Reweighting in Federated Learning

Based off of the 'Local reweighing' approach of [Abay et al 2020](https://arxiv.org/pdf/2012.02447). The following steps were performed:

1.  Create two datasets (`prepare_for_fl.Rmd`)
    1.  An 'overall' dataset from MEPS 19
    2.  Three 'sites' for federated learning. Each site contained roughly a third of the data.
2.  Using the [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) framework, fit a basic neural network both with/without re-weighting:
    1.  To all datasets separately: `meps_local_run.ipynb`
    2.  Using simulated federated learning with NVFLARE:
        1.  Standard run: `meps_nvflare_simulation.ipynb`
        2.  Weighted: `meps_nvflare_weighted_simulation.ipynb`
3.  Evaluate the results: `evaluate_nvflare.ipynb`
