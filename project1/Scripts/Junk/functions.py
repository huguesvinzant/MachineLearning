# -*- coding: utf-8 -*-
"""Functions for project 1."""
import numpy as np

def meaningless_to_nan(tx):
    """Remove meaningless -999 values (as explained in guidelines) 
        from training dataset by converting them into NaN."""
    tx[tx == -999] = np.NaN
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

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    error = y - pred
    loss = (((y - pred)**2).mean(axis = 0)) / 2
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = np.transpose(tx) @ (pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    return loss, (w - gamma * grad)

def logistic_regression_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []
    
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for i in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if i % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=i, l=loss))
        # converge criterion
        losses.append(loss)
        if i == 0:
            diff = losses[0]
        else:
            diff = np.abs(losses[-1] - losses[-2])
        if len(losses) > 1 and diff < threshold:
            break
    
    return losses, w
    # visualization
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    #print("loss={l}".format(l=calculate_loss(y, tx, w)))