# -*- coding: utf-8 -*-

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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """Compute the cost by negative log-likelihood."""
    pred = sigmoid(tx.dot(w))
    error = y - pred
    loss = (((y - pred)**2).mean(axis = 0)) / 2
    return loss

def calculate_gradient(y, tx, w):
    """Compute the gradient of loss for sigmoidal prediction."""
    pred = sigmoid(tx.dot(w))
    grad = np.transpose(tx) @ (pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    return loss, (w - gamma * grad)
def learning_by_gradient_descent_reg(y, tx, w, gamma,lambda_):
    """
    Do one step of gradient descent using regularized logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    return loss, (w - gamma * grad - gamma * lambda_*w)

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
    """Split the dataset based on the split ratio."""
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
    """Ridge regression built-in demo for trials."""
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

def logistic_regression(y, x, initial_w,max_iters, gamma):
    """
    Does one step of gradient descent using logistic regression. 
    Return the loss and the updated weight w.
    """
    threshold = 1e-8
    losses = []

    # build tx including w_0 weight
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))
    initial_w = w
    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if i % 10 == 0:
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
   
def reg_logistic_regression(y, x, lambda_, initial_w,max_iters, gamma):
    """
    Do one step of gradient descent using reg_logistic regression.
    Return the loss and the updated w.
    """
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))
    initial_w = w
    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_reg(y, tx, w, gamma,lambda_)
        # log info
        if i % 10 == 0:
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

def logistic_regression_gradient_descent_demo(y, x): # this should work
    """Logistic regression built-in demo for trials."""
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

def cross_validation(y, x, k_indices, k_fold, lambda_, degree=1):
    """Return the loss of ridge regression."""
    loss_tr = []
    loss_te = []
    initial_w = np.zeros((len(x[0]),1))
    max_iters = 1
    gamma = 0.01
    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        y_te = y[k_indices[k]]
        x_te = x[k_indices[k]]
        list_tr = []
        for l in range(k_fold):
            if l != k:
                list_tr.append(k_indices[l])
        y_tr = y[np.concatenate(list_tr)]
        x_tr = x[np.concatenate(list_tr)]
        # form data with polynomial degree
        #poly_tr = build_poly(x_tr, degree)
        poly_tr = x_tr
        #poly_te = build_poly(x_te, degree)
        poly_te = x_te
        # ridge regression
        w, _ = ridge_regression(y_tr, poly_tr, lambda_)
        #y_pred = np.dot(poly_te, w)
        #score = accuracy(y_pred,y_te)
        #w, _   = reg_logistic_regression(y_tr,poly_tr, lambda_, initial_w ,max_iters,gamma)
        # calculate the loss for train and test data
        loss_tr.append(compute_loss(y_tr, poly_tr, w))
        loss_te.append(compute_loss(y_te, poly_te, w))
        
    return np.mean(loss_tr),np.mean(loss_te)

def accuracy (y_pred,y):
    prop = 0
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            prop += 1
    return prop/len(y)

def class_accuracy(y_pred, labels):
    count_signal = len(np.extract(labels[:] == 1, labels))
    count_bg = len(np.extract(labels[:] == -1, labels))
    selection_signal = np.extract( np.logical_and(y_pred[:] != 1, labels[:] == 1), y_pred)
    selection_bg = np.extract( np.logical_and(y_pred[:] != -1, labels[:] == -1), y_pred)
    class_error = (1/3) * (len(selection_signal) / count_signal)
    class_error += (2/3) * (len(selection_bg) / count_bg) 
    class_score = 1 - class_error
    return class_score
                                                                                                         
def cross_validation_(y, x, k_indices, k_fold, lambda_, degree):
    """Return the loss of ridge regression."""
    loss_tr = []
    loss_te = []
    initial_w = np.zeros((len(x[0]),1))
    max_iters = 1
    gamma = 0.01
    for k in range(k_fold):
        # get k'th subgroup in test, others in train
        y_te = y[k_indices[k]]
        x_te = x[k_indices[k]]
        list_tr = []
        for l in range(k_fold):
            if l != k:
                list_tr.append(k_indices[l])
        y_tr = y[np.concatenate(list_tr)]
        x_tr = x[np.concatenate(list_tr)]
        #form data with polynomial degree
        poly_tr = build_poly(x_tr, degree)
        poly_te = build_poly(x_te, degree)
      
        # ridge regression
        w, _ = ridge_regression(y_tr, poly_tr, lambda_)
        y_pred = predict_labels(w, poly_te)
        score = accuracy(y_pred, y_te)
        class_score = class_accuracy(y_pred, y_te)
        #w, _   = reg_logistic_regression(y_tr,poly_tr, lambda_, initial_w ,max_iters,gamma)
        # calculate the loss for train and test data
        loss_tr.append(compute_loss(y_tr, poly_tr, w))
        loss_te.append(compute_loss(y_te, poly_te, w))
        
    return np.mean(loss_tr),np.mean(loss_te), score, class_score
        
def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def nan_find_columns(n_features,n_samples,nan_data):
    features = []
    for feature in range(n_features):
        for sample in range(n_samples):
            if np.isnan(nan_data[sample,feature]):
                features.append(feature)
    nan_columns = np.unique(features)
    print(nan_columns)   
    return nan_columns
