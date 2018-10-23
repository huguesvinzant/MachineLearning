import numpy as np

from proj1_helpers import *

def get_jet_indices(labels, raw_data):
    """Get the sample idices of each number of jets"""
    jet_0 = []
    jet_1 = []
    jet_2 = []
    for i in range(len(labels)):
        if raw_data[i,23] == -999:
            jet_0.append(i)
        else:
            if raw_data[i,4] == -999:
                jet_1.append(i)
            else:
                jet_2.append(i)
    return np.asarray(jet_0), np.asarray(jet_1), np.asarray(jet_2)
                
def divide_data(labels, raw_data):
    """Divide the data according to the number of jets"""
    jet0, jet1, jet2 = get_jet_indices(labels, raw_data)
    
    jet0_labels = labels[jet0]
    jet0_data = raw_data[jet0,:]
    jet0_data[jet0_data == -999] = 0

    jet1_labels = labels[jet1]
    jet1_data = raw_data[jet1,:]
    jet1_data[jet1_data == -999] = 0

    jet2_labels = labels[jet2]
    jet2_data = raw_data[jet2,:]
    jet2_data[jet2_data == -999] = 0
    
    return jet0_labels, jet0_data, jet1_labels, jet1_data, jet2_labels, jet2_data

def standardize_train(train_data):
    """Standardize the train data along the feature axis."""
    mean_data = np.mean(train_data, axis = 0)
    centered_data = train_data - mean_data
    std_data = np.std(centered_data, axis = 0)
    std_data[std_data == 0] = 1
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
    y_pred = predict_labels(w, tx)
    y_pred[y_pred == -1] = 0
    y[y == -1] = 0
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    return mse

def column_estimation_train(data):
    feature = 0
    n_samples, n_features = np.shape(data)
    data[data == -999] = np.NaN
    
    submatrix = np.delete(data, feature, axis = 1)

    samples = []
    for sample in range(n_samples):
        if np.isnan(data[sample,feature]):
            samples.append(sample)
        nan_lines = np.unique(samples)
    
    submatrix0 = np.delete(submatrix, nan_lines, axis = 0)
    labels0 = np.delete(data[:,feature], nan_lines, axis = 0)
        
    weights_train, _ = least_squares(labels0, submatrix0)
    x_pred = np.dot(submatrix[nan_lines,:], weights_train)
    data[nan_lines, feature] = x_pred
        
    return data, weights_train

def column_estimation_test(nan_data,nan_columns,weights_train):
    n_samples, n_features = np.shape(nan_data)
    submatrix = np.delete(nan_data, nan_columns, axis = 1)
    for f_ind, chosen_feature in enumerate(nan_columns):
        samples = []
        for sample in range(n_samples):
            if np.isnan(nan_data[sample,chosen_feature]):
                samples.append(sample)
        nan_lines = np.unique(samples)
        x_pred = np.dot(submatrix[nan_lines,:], weights_train[f_ind,:])
        nan_data[nan_lines, chosen_feature] = x_pred
        
    return nan_data