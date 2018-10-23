import numpy as np

from proj1_helpers import *

def meaningless_to_nan(tx):
    """Remove meaningless -999 values (as explained in guidelines) 
        from training dataset by converting them into NaN."""
    tx[tx == -999] = np.NaN
    return tx

def nan_find_columns(nan_data):
    """Find columns contaning some NaN."""
    n_samples, n_features = np.shape(nan_data)
    features = []
    for feature in range(n_features):
        for sample in range(n_samples):
            if np.isnan(nan_data[sample,feature]) and feature != 0:
                features.append(feature)
    nan_columns = np.unique(features)
    print('Columns containing NaN', nan_columns)   
    return nan_columns
                
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

def standardize_train(train_data):
    """Standardize the train data along the feature axis."""
    mean_data = np.nanmean(train_data, axis = 0)
    centered_data = train_data - mean_data
    std_data = np.nanstd(centered_data, axis = 0)
    standardized_data = centered_data / std_data
    return standardized_data, mean_data, std_data

def standardize_test(test_data, mean_train, std_train):
    """Standardize the test data along the feature axis."""
    standardized_data_te = (test_data - mean_train) / std_train
    return standardized_data_te

def least_squares(y, tx):
    """Calculate the least squares solution using normal equations."""
    inverse = np.linalg.inv(np.transpose(tx)@tx)
    weights = inverse@np.transpose(tx)@y
    loss = compute_loss(y, tx, weights)
    return weights, loss

def compute_loss(y, tx, w):
    """Calculate the mse loss."""
    y_pred = tx.dot(w)
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    return mse

def column_estimation_train(nan_data):
    chosen_feature = 0
    submatrix = np.delete(nan_data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    for sample in range(n_samples):
        if np.isnan(nan_data[sample,chosen_feature]):
            samples.append(sample)
    nan_lines = np.unique(samples)
    submatrix_bis = np.delete(submatrix, nan_lines, axis = 0)
    labels_bis = np.delete(nan_data[:,chosen_feature], nan_lines, axis = 0)
    weights_train, _ = least_squares(labels_bis, submatrix_bis)
    x_pred = np.dot(submatrix[nan_lines,:], weights_train)
    nan_data[nan_lines, chosen_feature] = x_pred
        
    return nan_data,weights_train

def column_estimation_test(nan_data, weights_train):
    chosen_feature = 0
    submatrix = np.delete(nan_data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    for sample in range(n_samples):
        if np.isnan(nan_data[sample,chosen_feature]):
            samples.append(sample)
    nan_lines = np.unique(samples)
    submatrix_bis = np.delete(submatrix, nan_lines, axis = 0)
    labels_bis = np.delete(nan_data[:,chosen_feature], nan_lines, axis = 0)
    
    x_pred = np.dot(submatrix[nan_lines,:], weights_train)
    nan_data[nan_lines, chosen_feature] = x_pred
        
    return nan_data

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k_fold, lambda_, degree):
    """Cross-validation with the loss of ridge regression."""
    loss_tr = []
    loss_te = []
    max_iters = 10
    initial_w = np.zeros((len(x[0]),1))
    #gamma = 0.01
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
        w, loss_tr = ridge_regression(y_tr, poly_tr, lambda_)
        y_pred = predict_labels(w, poly_te)
        score = accuracy(y_pred, y_te)
        
        # calculate the loss for test data
        loss_te.append(compute_loss_labels(y_te, poly_te, w))
        
    return np.mean(loss_tr),np.mean(loss_te), score

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

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

def compute_loss_labels(y, tx, w):
    """Calculate the mse loss."""
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

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred