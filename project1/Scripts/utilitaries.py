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

def remove_meaningless_data(tx):
    """ Remove meaningless -999 values from training dataset."""
    tx[tx == -999] = np.NaN
    return tx

def standardize(x):
    mean =  np.nanmean(x,0) #dim D | or center the data directly : centered_data = x - np.mean(x,axis=0)
    std = np.nanstd(x,0) 
    return (x - mean) / std
    