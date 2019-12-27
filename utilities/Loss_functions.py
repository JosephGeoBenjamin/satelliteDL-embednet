import torch
import torch.nn as nn



def distributional_CE(soft_pred, soft_targets):
    """ For applying cross-entropy over distribution
    Both inputs are soft i.e 0-1 Prob Distribution
    """
    return torch.mean(torch.sum(- soft_targets * torch.log(pred), 1))