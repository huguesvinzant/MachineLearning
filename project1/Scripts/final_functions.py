import numpy as np
from proj1_helpers import *

************************************************************
********************** Implementation **********************
************************************************************

#GD
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

#SGD
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

#Least-squares
def least_squares(y, tx):
    """Calculate the least squares solution using normal equations."""
    inverse = np.linalg.inv(np.transpose(tx)@tx)
    weights = inverse@np.transpose(tx)@y
    loss = compute_loss(y, tx, weights)
    return weights, loss

#Ridge regression
def ridge_regression(y, tx, lambda_):
    """Explicit solution for the weights using ridge regression."""
    lambda_prime = lambda_ * 2 * len(y)
    # compute explicit solution for the weights
    a = np.transpose(tx) @ tx + lambda_prime * np.identity(tx.shape[1])
    b = np.transpose(tx) @ y
    weights = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, weights)
    return weights, loss

#Logistic regression

#Regularized logistic regression

************************************************************
*********************** Utilitaries ***********************
************************************************************

def compute_loss(y, tx, w):
    """Calculate the mse loss."""
    y_pred = tx.dot(w)
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    return mse

def compute_loss_labels(y, tx, w):
    """Calculate the mse loss on the computed labels."""
    y_pred = predict_labels(w, tx)
    y_pred[y_pred == -1] = 0
    y[y == -1] = 0
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    return mse

def accuracy (y_pred,y):
    """Compute accuracy."""
    prop = 0
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            prop += 1
    return prop/len(y)

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k_fold, lambda_, degree): #rename with ridge????
    """Cross-validation with the loss of ridge regression."""
    score = []
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
        score.append(accuracy(y_pred, y_te))
    return np.mean(score)

def find_best_parameters(labels, data, k_fold, lambdas, degrees, seed): #rename with ridge????
    """Find the best parameters for ridge regression based on cross validation."""
    k_idx = build_k_indices(labels, k_fold, seed)
    loss_te = np.ones((len(degrees),len(lambdas)))
    scores = np.ones((len(degrees),len(lambdas)))
    for degree_idx, degree in enumerate(degrees):
        for lambda_idx, lambda_ in enumerate(lambdas):
            scores[degree_idx, lambda_idx]= cross_validation(labels, data, k_idx, k_fold, lambda_, degree)
    best_HP_idx = np.unravel_index(np.argmax(scores), np.shape(scores))
    best_degree = degrees[best_HP_idx[0]]
    best_lambda = lambdas[best_HP_idx[1]]
    best_score = scores[best_HP_idx[0], best_HP_idx[1]]
    print('Best degree:', best_degree, 'Best lambda:', best_lambda, 'Best score:', best_score)
    return best_degree, best_lambda

def make_predictions(estimated_data, labels, estimated_data_te, best_lambda, best_degree):
    """Use the best parameters to make the prediction"""
    poly_data = build_poly(estimated_data, best_degree)
    weights, loss = ridge_regression(labels, poly_data, best_lambda)
    poly_data_te = build_poly(estimated_data_te, best_degree)
    y_pred = predict_labels(weights, poly_data_te)
    return y_pred

************************************************************
********************* Data processing **********************
************************************************************

def divide_data(labels, raw_data):
    """Divide the data according to the number of jets"""
    # information of jets given by the 22th feature
    jet0_data = raw_data[np.where(raw_data[:,22]== 0)[0],:]
    jet0_labels = labels[np.where(raw_data[:,22]== 0)[0]]
    jet1_data = raw_data[np.where(raw_data[:,22]== 1)[0],:]
    jet1_labels = labels[np.where(raw_data[:,22]== 1)[0]]
    jet2_data = raw_data[np.where(raw_data[:,22] > 1)[0],:]
    jet2_labels = labels[np.where(raw_data[:,22] > 1)[0]]
    return jet0_labels, jet0_data, jet1_labels, jet1_data, jet2_labels, jet2_data

def find_null_variance_features(data):
    """Find for each jet the features with a variance equal to 0"""
    variance = np.var(data, axis = 0)
    no_var = np.where(variance == 0)
    print('Columns with variance = 0 :', no_var)
    return no_var

def remove_novar_features(data, data_te):
    """Remove features with a variance equal to 0"""
    no_var_columns = find_null_variance_features(data)
    clean_data = np.delete(data, no_var_columns, axis = 1)
    clean_data_te = np.delete(data_te, no_var_columns, axis = 1)
    print('New data shape :', clean_data.shape)
    return clean_data, clean_data_te

def meaningless_to_nan(tx):
    """Remove meaningless -999 values by converting them into NaN."""
    tx[tx == -999] = np.NaN
    return tx

def standardize_train(train_data):
    """Standardize the train data along the feature axis."""
    train_data = meaningless_to_nan(train_data)
    mean_data = np.nanmean(train_data, axis = 0)
    centered_data = train_data - mean_data
    std_data = np.nanstd(centered_data, axis = 0)
    standardized_data = centered_data / std_data
    return standardized_data, mean_data, std_data

def standardize_test(test_data, mean_train, std_train):
    """Standardize the test data along the feature axis."""
    test_data = meaningless_to_nan(test_data)
    standardized_data_te = (test_data - mean_train) / std_train
    return standardized_data_te

def column_estimation_train(data):
    """Estimate the NaN in column 0 based on the other features"""
    chosen_feature = 0
    degree = 1
    submatrix = np.delete(data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    for sample in range(n_samples):
        if (np.isnan(data[sample,chosen_feature])):
            samples.append(sample)
    print(len(samples), 'NaN lines found')
    submatrix_bis = np.delete(submatrix, samples, axis = 0)
    labels_bis = np.delete(data[:,chosen_feature], samples, axis = 0)
    poly = build_poly(submatrix_bis, degree)
    weights_train, _ = least_squares(labels_bis, poly)
    poly_te = build_poly(submatrix[samples,:], degree)
    x_pred = np.dot(poly_te, weights_train)
    data[samples, chosen_feature] = x_pred
    return data, weights_train

def column_estimation_test(data, weights_train):
    """"Estimate the NaN in column 0 based on the other features adn the train weights"""
    chosen_feature = 0
    degree = 1
    submatrix = np.delete(data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    for sample in range(n_samples):
        if (np.isnan(data[sample,chosen_feature])):
            samples.append(sample)
    submatrix_bis = np.delete(submatrix, samples, axis = 0)
    labels_bis = np.delete(data[:,chosen_feature], samples, axis = 0)
    poly_te = build_poly(submatrix[samples,:], degree)
    x_pred = np.dot(poly_te, weights_train)
    data[samples, chosen_feature] = x_pred
    return data