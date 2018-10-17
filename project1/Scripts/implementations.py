# -*- coding: utf-8 -*-
"""Working toolbox with 6 basic functions."""
import matplotlib.pyplot as plt
from utilitaries import *

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

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")

def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

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
            rmse_tr.append(np.sqrt(2 * loss_tr))
            rmse_te.append(np.sqrt(2 * compute_loss(y_te, tx_te, weight_tr)))
            
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w = initial_w
    losses = []
    weights =[]
    for step in range(max_iters):
        loss = calculate_logistic_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w -= gamma * grad
        if step % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=step, l=loss))
        losses.append(loss)
        weights.append(w)
    return losses, weights

def reg_logistic_regression(y, tx, lambda_, initial_w,max_iters, gamma):
    """
    Do one step of gradient descent using reg_logistic regression.
    Return the loss and the updated w.
    """
    w = initial_w
    losses = []
    weights =[]
    for step in range(max_iters):
        for i in range(tx.shape[0]):
            grad = calculate_gradient(y, tx, w)
        print(grad.shape)
        w -= gamma * grad - gamma * lambda_*w
        weights.append(w)
        if step % 100 == 0:
            print('Current iteration={i}:', step, 'loss={l}:', loss)
        loss = calculate_logistic_loss(y, tx, w)
        losses.append(loss)
        
    return losses, weights

def logistic_regression_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []
    # build tx
    #tx = np.c_[np.ones((y.shape[0], 1)),x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for i in range(max_iter):
        # get loss and update w.
        loss, w = logistic_regression_by_gradient_descent(y, tx, w, gamma)
        # log info
        if i % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))



