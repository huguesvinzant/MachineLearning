# -*- coding: utf-8 -*-
"""Working toolbox with 6 basic functions."""
from utilitaries import *

def least_squares(y, tx):
    """Calculate the least squares solution using normal equations."""
    inverse = np.linalg.inv(np.transpose(tx)@tx)
    weights = inverse@np.transpose(tx)@y
    losses = compute_loss(y, tx, weights)
    return weights, losses


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Calculate linear regression using gradient descent."""
    # Define parameters to store weights and loss
    weights = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        weights.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return weights, losses


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Calculate linear regression using stochastic gradient descent."""
    # Define parameters to store weights and loss
    weights = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            weights.append(w)
            losses.append(loss)

       # print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
       #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return weights, losses


def ridge_regression(y, tx, lambda_)
    ''' '''
    
    return weights, losses