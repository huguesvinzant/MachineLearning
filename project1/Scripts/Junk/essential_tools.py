import numpy as np

from proj1_helpers import *

def meaningless_to_nan(tx):
    """Remove meaningless -999 values (as explained in guidelines) 
        from training dataset by converting them into NaN."""
    tx[tx == -999] = np.NaN
    return tx

def find_null_variance_features(data):
    variance = np.var(data, axis = 0)
    no_var = np.where(variance == 0)
    print('Columns with variance = 0 :', no_var)
    return no_var
def compute_loss(y, tx, w):
    """Calculate the mse loss."""
    y_pred = tx.dot(w)
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    return mse

def remove_novar_features(data, data_te):
    no_var_columns = find_null_variance_features(data)
    clean_data = np.delete(data, no_var_columns, axis = 1)
    clean_data_te = np.delete(data_te, no_var_columns, axis = 1)
    print('New data shape :', clean_data.shape)
    return clean_data, clean_data_te
                
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

def least_squares(y, tx):
    """Calculate the least squares solution using normal equations."""
    inverse = np.linalg.inv(np.transpose(tx)@tx)
    weights = inverse@np.transpose(tx)@y
    loss = compute_loss(y, tx, weights)
    return weights, loss


def column_estimation_train(data):
    chosen_feature = 0
    nan_found = True
    submatrix = np.delete(data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    for sample in range(n_samples):
        if (np.isnan(data[sample,chosen_feature])):
            samples.append(sample)
    print(len(samples), 'NaN lines found')
    submatrix_bis = np.delete(submatrix, samples, axis = 0)
    labels_bis = np.delete(data[:,chosen_feature], samples, axis = 0)
    weights_train, _ = least_squares(labels_bis, submatrix_bis)
    x_pred = np.dot(submatrix[samples,:], weights_train)
    data[samples, chosen_feature] = x_pred
    
    return data, weights_train

def column_estimation_test(data, weights_train):
    chosen_feature = 0
    submatrix = np.delete(data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    for sample in range(n_samples):
        if (np.isnan(data[sample,chosen_feature])):
            samples.append(sample)
    submatrix_bis = np.delete(submatrix, samples, axis = 0)
    labels_bis = np.delete(data[:,chosen_feature], samples, axis = 0)
    
    x_pred = np.dot(submatrix[samples,:], weights_train)
    data[samples, chosen_feature] = x_pred
        
    return data


def find_best_parameters(labels, data, k_fold, lambdas, degrees, seed):
    k_idx = build_k_indices(labels, k_fold, seed)
    loss_te = np.ones((len(degrees),len(lambdas)))
    scores = np.ones((len(degrees),len(lambdas)))
    
    for degree_idx, degree in enumerate(degrees):
        for lambda_idx, lambda_ in enumerate(lambdas):
            _ ,loss_te[degree_idx, lambda_idx], scores[degree_idx, lambda_idx]= cross_validation(labels, data, k_idx, k_fold, lambda_, degree)
            #print('Degree:', degrees[degree_idx], 'Lambda:', lambdas[lambda_idx])
            #print('Score:', scores[degree_idx, lambda_idx])
            #print('Loss:', loss_te[degree_idx, lambda_idx])
    
    ratio = scores/loss_te
    best_HP_idx = np.unravel_index(np.argmax(scores), np.shape(scores))
    best_degree = degrees[best_HP_idx[0]]
    best_lambda = lambdas[best_HP_idx[1]]
    best_score = scores[best_HP_idx[0], best_HP_idx[1]]
    best_loss = loss_te[best_HP_idx[0], best_HP_idx[1]]
    print('Best degree:', best_degree, 'Best lambda:', best_lambda, 'Best score:', best_score, 'Best loss:', best_loss)
    
    return best_degree, best_lambda

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
    #initial_w = np.zeros((len(x[0]),1))
    gamma = 1e-07
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
      
        # logistic regression
        initial_w = np.zeros((poly_tr.shape[1],1))
   
        #w,loss_tr = logistic_regression(y_tr, poly_tr, initial_w,max_iters, lambda_)
        w,loss_tr = reg_logistic_regression(y_tr, poly_tr, lambda_, initial_w, max_iters, gamma)
        
        y_pred = final_predict_labels_log(w,poly_te)
        score = accuracy(y_pred,y_te)
        
        # calculate the loss for test data
        loss_te.append(calculate_loss_(y_te, poly_te, w))
        
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
    y_pred = final_predict_labels_log(w, tx)
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


def make_predictions(estimated_data, labels, estimated_data_te, best_lambda, best_degree):
    
    max_iters = 2
    poly_data = build_poly(estimated_data, best_degree)
    initial_w = np.zeros((len(poly_data[0]),1))
    gamma = 1e-06
    #weights, loss = ridge_regression(labels, poly_data, best_lambda)
    loss, weights = reg_logistic_regression(labels, poly_data, best_lambda, initial_w,max_iters,gamma)
    poly_data_te = build_poly(estimated_data_te, best_degree)
    y_pred = final_predict_labels_log(weights, poly_data_te)
    return y_pred

## -----LOGISTIC FUNCTIONS + TOOLS NEW FUNCTIONS --------------------------------------------------------------------------------
## TO ADD:
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

def final_predict_labels_log(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data,weights)
    y_pred = sigmoid(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    print('y_pred shape', y_pred.shape)
    return y_pred

def predict_labels_log(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred = sigmoid(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """Compute the cost by negative log-likelihood."""
    pred = sigmoid(tx.dot(w))
    correction_factor = 1e-10;
    loss = y.T.dot(np.log(pred + correction_factor)) + (1.0 - y).T.dot(np.log((1.0 - pred)+ correction_factor))
    return np.squeeze(-loss)

def calculate_logistic_gradient(y, tx, w):
    """Compute the gradient of loss for sigmoidal prediction."""
    pred = sigmoid(tx.dot(w))
    err = pred - y
    grad = np.transpose(tx) @ err
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """Do one step of gradient descent using logistic regression."""
    loss = calculate_logistic_loss(y, tx, w)
    grad = calculate_logistic_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def logistic_regression(y, x, initial_w,max_iters, gamma):
    """Does one step of gradient descent using logistic regression."""
    w = initial_w
    update_tol = 1e-6
    loss = 0
    losses=np.array([])
    for iter in range(max_iters):
        previous_loss = loss
        loss, w = learning_by_gradient_descent(y,x, w, gamma)
        if iter%30 == 0:
            if np.abs((previous_loss - loss)/loss) < update_tol:
                gamma = 0.8*gamma
    return w,loss

def learning_by_penalized_gradient(y, tx, w, gamma,lambda_):
    """Do one step of gradient descent using regularized logistic regression."""
    loss = calculate_logistic_loss(y, tx, w)+ lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    new_w = w - gamma * grad
    return loss, new_w


def reg_logistic_regression(y, x, lambda_, initial_w,max_iters, gamma):
    """Do one step of gradient descent using reg_logistic regression."""
    w = initial_w
    update_tol = 1e-6
    loss = 0
    losses=np.array([])
    for iter in range(max_iters):
        previous_loss = loss
        loss, w = learning_by_penalized_gradient(y,x, w, gamma,lambda_)
        if iter%30 == 0:
            if np.abs((previous_loss - loss)/loss) < update_tol:
                gamma = 0.8*gamma
    return w,loss

def logistic_find_best_parameters(y, tx, lambdas, gamma, degrees):
    """Find the best parameters for logistic regression based on cross validation."""
    loss_tr = np.zeros((len(lambdas), len(degrees)))
    loss_te = np.zeros((len(lambdas), len(degrees)))
    accuracy = np.zeros((len(lambdas), len(degrees)))
    for lambda_idx, lambda_ in enumerate(lambdas):
        for degree_idx, degree in enumerate(degrees):          
            poly_x = build_poly(tx, degree)
            k_fold = 10
            seed = 1
            k_indices = build_k_indices(y, k_fold, seed)
            mean_accuracy,mean_loss_te = cross_validation_log(y, poly_x, k_indices, lambda_, gamma)        
            accuracy[lambda_idx, degree_idx] = mean_accuracy
            loss_te[lambda_idx, degree_idx] = mean_loss_te
            print('Degree:', degrees[degree_idx], 'Lambda:', lambdas[lambda_idx])
    ratio = accuracy/loss_te
    best_HP_idx = np.unravel_index(np.argmax(ratio), np.shape(ratio))
    best_degree = degrees[best_HP_idx[1]]
    best_lambda = lambdas[best_HP_idx[0]]
    best_accuracy = accuracy[best_HP_idx[0],best_HP_idx[1]]
    best_loss = loss_te[best_HP_idx[0],best_HP_idx[1]]
    return best_lambda, best_degree,best_accuracy,best_loss

def cross_validation_log(y, x, k_indices, gamma, lambda_):
    kfolds = k_indices.shape[0]
    accuracy = np.zeros(kfolds)
    loss_te = np.zeros(kfolds)
    if len(x.shape) == 1:
        w = 0
    else:
        w = np.zeros(x.shape[1]) 
    for k in range(kfolds):
        idx = k_indices[k]
        yte = y[idx]
        if len(x.shape) == 1:
            xte = x[idx]
        else:
            xte = x[idx,:]
        ytr = np.delete(y,idx,0)
        xtr = np.delete(x,idx,0)
        loss_te[k], w = learning_by_penalized_gradient(ytr, xtr, w, gamma, lambda_)
        #accuracy
        y_pred = predict_labels_log(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)  
    return np.mean(accuracy), np.mean(loss_te)
