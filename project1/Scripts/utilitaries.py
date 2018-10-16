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
    """Remove meaningless -999 values (as explained in guidelines) 
        from training dataset by converting them into NaN."""
    tx[tx == -999] = np.NaN
    return tx

def nan_proportion(tx):
    """Calculate and display the proportion [0,1] of NaN per feature (column)."""
    n_samples, n_features = np.shape(tx)
    #init. matrix that will contain proportions
    prop_nan = np.zeros(30)
    remove_indices = []
    for feature in range(n_features):
        n_nan = 0
        for sample in range(n_samples):
            if np.isnan(tx[sample,feature]):
                n_nan += 1
        prop_nan[feature] = n_nan/n_samples
    return prop_nan, n_features


def remove_meaningless_features(tx, ratio):
    """Remove features with more than 'ratio' of NaN."""
    tx = meaningless_to_nan(tx)
    prop_nan, n_features = nan_proportion(tx)
    remove_indices = []
    for i in range(n_features):
        if prop_nan[i] > ratio:
            remove_indices.append(i)
    print('Removed features :',remove_indices)
    tx = np.delete(tx, remove_indices, axis=1)
    return tx

def standardize(tx):
    """Standardize the data along the feature_wise ignoring NaN entries."""
    mean_tx = np.nanmean(tx, axis = 0)
    centered_tx = tx - mean_tx
    centered_tx[np.isnan(centered_tx)] = 0
    std_tx = np.nanstd(centered_tx,axis = 0)
    standardized_tx = centered_tx / std_tx
    return standardized_tx

def PCA(tx, t):
    """Apply PCA to a given set of datapoints in D-dimension."""
    cov_matrix = np.cov(tx.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    
    # Top k eigenvectors bear the most information about the data distribution
    sort_indices = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sort_indices]
    eigen_vectors = eigen_vectors[:,sort_indices]
    eigenval = np.asarray(eigen_values)
    eigenvec = np.asarray(eigen_vectors)
    total = sum(eigenval)
    
    explained_variance =[]   #how much information can be attributed to each of the principal component
    k_feature = 0
    sum_explained_var = 0
    for i in eigenval:
        explained_variance.append([(i/total)*100])
        sum_explained_var += (i/total)
        if sum_explained_var < t: 
            k_feature += 1  
    print('Kept features:', k_feature)
    return eigen_values, eigen_vectors[:,:k_feature], explained_variance