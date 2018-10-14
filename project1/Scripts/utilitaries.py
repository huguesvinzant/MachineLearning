# -*- coding: utf-8 -*-
"""Utilitary functions."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the mse loss."""
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def meaningless_to_nan(tx):
    """Remove meaningless -999 values (as explained in the guidelines) 
        from training dataset by converting them into NaN."""
    tx[tx == -999] = np.NaN
    return tx

def nan_proportion(tx):
    """Calculate and display the proportion [0,1] of NaN per feature (column)."""
    n_samples = tx.shape[0]
    n_features = tx.shape[1]
    #init. matrix that will contain proportions
    prop_nan = np.zeros((2,30))
    
    for feature in range(n_features):
        n_nan = 0
        for sample in range(n_samples):
            if np.isnan(tx[sample,feature]):
                n_nan += 1
                prop_nan[0,feature] = feature
                prop_nan[1,feature] = n_nan/n_samples
    return prop_nan

def standardize(x):
    """Standardize the data feature-wise ignoring NaN entries."""
    mean =  np.nanmean(x,0) #dim D | or center the data directly : centered_data = x - np.mean(x,axis=0)
    std = np.nanstd(x,0) 
    return (x - mean) / std
    