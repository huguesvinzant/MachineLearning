# -*- coding: utf-8 -*-
"""Working toolbox with 6 basic functions."""
from utilitaries import *
from additional_functions import *

def least_squares(y, tx):
    """Calculate the least squares solution using normal equations."""
    inverse = np.linalg.inv(np.transpose(tx)@tx)
    weights = inverse@np.transpose(tx)@y
    loss = compute_loss(y, tx, weights)
    return weights, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Calculate linear regression using gradient descent."""
    # Define parameters 
    weights = initial_w
    for n_iter in range(max_iters):
        # compute loss and gradient
        grad, err = compute_gradient(y, tx, weights)
        loss = calculate_mse(err)
        # gradient weights through descent update
        weights = weights - gamma * grad
    return weights, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):  # does not seem to work
    """Calculate linear regression using stochastic gradient descent."""
    # Define parameters 
    weights = initial_w 
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, weights)
            # update weights through the stochastic gradient update
            weights = weights - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, weights)
    return weights, loss


def ridge_regression(y, tx, lambda_):
    '''Explicit solution for the weights using ridge regression.'''
    # compute another lambda to simplify notation
    lambda_prime = lambda_ * 2 * len(y)
    # compute explicit solution for the weights
    a = np.transpose(tx) @ tx + lambda_prime * np.identity(tx.shape[1])
    b = np.transpose(tx) @ y
    weights = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, weights)
    return weights, loss

def ridge_regression_demo(std_data,labels,degree,ratio,seed):
    """Ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, -1, 10)
    # split data
    x_tr, x_te, y_tr, y_te = split_data(std_data,labels,ratio, seed)
    # form tx
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression with different lambda
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
         # ridge regression
            weight_tr,loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
            weight_te,loss_te = ridge_regression(y_te, tx_te, lambda_)
            rmse_tr.append(np.sqrt(2 * loss_tr))
            rmse_te.append(np.sqrt(2 * loss_te))
            
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)
