import numpy as np

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
            if np.isnan(nan_data[sample,feature]):
                features.append(feature)
    nan_columns = np.unique(features)
    print('Columns containing NaN', nan_columns)   
    return nan_columns

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k_fold, lambda_):
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
        # ridge regression
        w, _ = ridge_regression(y_tr, x_tr, lambda_)
        # calculate the loss for train and test data
        loss_tr.append(compute_loss(y_tr, x_tr, w))
        loss_te.append(compute_loss(y_te, x_te, w))
        
    return np.mean(loss_tr),np.mean(loss_te)

def column_estimation(nan_data):
    
    n_samples, n_features = np.shape(nan_data)
    nan_columns = nan_find_columns(nan_data)
    submatrix = np.delete(nan_data, nan_columns, axis = 1)

    for chosen_feature in nan_columns:
        samples = []
        for sample in range(n_samples):
            if np.isnan(nan_data[sample,chosen_feature]):
                samples.append(sample)
        nan_lines = np.unique(samples)
        submatrix0 = np.delete(submatrix, nan_lines, axis = 0)
        labels0 = np.delete(nan_data[:,chosen_feature], nan_lines, axis = 0)
        
        weights_c, _ = least_squares(labels0, submatrix0)
        x_pred = np.dot(submatrix[nan_lines,:], weights_c)
        nan_data[nan_lines, chosen_feature] = x_pred
        
    return nan_data

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

def compute_loss(y, tx, w):
    """Calculate the mse loss."""
    y_pred = predict_labels(w, tx)
    y_pred[y_pred == -1] = 0
    y[y == -1] = 0
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    return mse

def cross_validation_(y, x, k_indices, k_fold, lambda_, degree):
    """Cross-validation with the loss of ridge regression."""
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
        w, loss_tr = ridge_regression(y_tr, poly_tr, lambda_)
        y_pred = predict_labels(w, poly_te)
        score = accuracy(y_pred, y_te)
        
        # calculate the loss for test data
        loss_te.append(compute_loss(y_te, poly_te, w))
        
    return np.mean(loss_tr),np.mean(loss_te), score

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

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def standardize_train(train_data):
    """Standardize the data along the feature axis."""
    mean_data = np.mean(train_data, axis = 0)
    centered_data = train_data - mean_data
    std_data = np.std(centered_data, axis = 0)
    standardized_data = centered_data / std_data
    return standardized_data, mean_data, std_data

def standardize_test(test_data, mean_train, std_train):
    standardized_data_te = (test_data - mean_train) / std_train
    return standardized_data_te

def least_squares(y, tx):
    """Calculate the least squares solution using normal equations."""
    inverse = np.linalg.inv(np.transpose(tx)@tx)
    weights = inverse@np.transpose(tx)@y
    loss = compute_loss(y, tx, weights)
    return weights, loss