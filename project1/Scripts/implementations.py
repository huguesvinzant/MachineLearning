import numpy as np
from proj1_helpers import *

''''***********************************************************
********************** Implementations *************************
************************************************************'''

#----GD
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Calculate linear regression using gradient descent."""
    
    # Define parameters 
    weights = initial_w
    
    for n_iter in range(max_iters):
        # compute loss and gradient
        grad, err = compute_gradient(y, tx, weights)
        # calculate loss
        loss = calculate_mse(err)
        # gradient weights through descent update
        weights = weights - gamma * grad
        
    return weights, loss


#----SGD
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


#----Least-squares
def least_squares(y, tx):
    """Calculate the least squares explicit solution using normal equations."""
    
    inverse = np.linalg.inv(np.transpose(tx) @ tx)
    weights = inverse @ np.transpose(tx)  @y
    loss = compute_loss(y, tx, weights)
    
    return weights, loss


#----Ridge regression
def ridge_regression(y, tx, lambda_):
    """Explicit solution for the weights using ridge regression."""
    #Define paramater
    lambda_prime = lambda_ * 2 * len(y)
    # compute explicit solution for the weights
    a = np.transpose(tx) @ tx + lambda_prime * np.identity(tx.shape[1])
    b = np.transpose(tx) @ y
    weights = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, weights)
    
    return weights, loss


#----Logistic regression
def logistic_regression(y, x, initial_w, max_iters, gamma):
    """Do one step of gradient descent using logistic regression."""
    
    #Define parameters
    weights = initial_w
    update_tol = 1e-6
    loss = 0
    losses=np.array([])
    
    for iter in range(max_iters):
        previous_loss = loss
        loss, weights = learning_by_gradient_descent(y, x, weights, gamma)
        if iter % 30 == 0:
            if np.abs((previous_loss - loss)/loss) < update_tol:
                gamma = 0.8*gamma
                
    return weights, loss

#----Regularized logistic regression
def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
    """Do one step of gradient descent using regularized logistic regression."""
    
    #Define parameters
    weights = initial_w
    update_tol = 1e-6
    loss = 0
    losses=np.array([])
    
    for iter in range(max_iters):
        previous_loss = loss
        loss, weights = learning_by_penalized_gradient(y, x, weights, gamma, lambda_)
        if iter % 30 == 0:
            if np.abs((previous_loss - loss)/loss) < update_tol:
                gamma = 0.8*gamma
                
    return weights, loss

''''***********************************************************
************************ Utilitaries **************************
************************************************************'''

def compute_loss(y, tx, w):
    """Calculate the MSE loss."""
    
    y_pred = tx.dot(w)
    e = y - y_pred
    
    return  1/2*np.mean(e**2)
    

def compute_loss_labels(y, tx, w):
    """Calculate the mse loss on the computed labels."""
    
    y_pred = predict_labels(w, tx)
    y_pred[y_pred == -1] = 0
    y[y == -1] = 0
    e = y - y_pred
    mse = 1/2*np.mean(e**2)
    
    return mse


def sigmoid(t):
    """Apply sigmoid function on t."""
    
    return 1.0 / (1 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """Compute the cost by negative log-likelihood."""
    
    pred = sigmoid(tx.dot(w))
    correction_factor = 1e-10;
    loss = y.T.dot(np.log(pred + correction_factor)) + (1.0 - y).T.dot(np.log((1.0 - pred)+ correction_factor))
    
    return np.squeeze(-loss) #removes single dimensional entries


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
    new_w -= gamma * grad
    
    return loss, new_w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """Do one step of gradient descent using regularized logistic regression."""
    
    loss = calculate_logistic_loss(y, tx, w)+ lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    new_w = w - gamma * grad
    
    return loss, new_w


def final_predict_labels_log(weights, data):
    """Generates class predictions given weights, and a test data matrix."""
    
    y_pred = np.dot(data,weights)
    y_pred = sigmoid(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def predict_labels_log(weights, data):
    """Generates class predictions given weights, and a test data matrix."""
    
    y_pred = np.dot(data, weights)
    y_pred = sigmoid(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def accuracy (y_pred,y):
    """Compute accuracy score."""
    
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
    """Build k indices for k-fold split."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k_fold, lambda_, degree): #rename with ridge????
    """Cross-validation for ridge regression."""
    
    score = []
    #k-fold
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

#trial single function
def cross_validation_general(y, x, k_indices, k_fold, **param): #make sure when you call param['...']
    """Cross-validation for ridge regression."""
    
    score = []
    loss_te = np.zeros(k_fold)
    
        #k-fold
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
        poly_tr = build_poly(x_tr, param['degree'])
        poly_te = build_poly(x_te, param['degree'])
    
       # logistic
        if 'gamma' in param:
            
            if len(x.shape) == 1:
                w = 0
            else:
                w = np.zeros(x.shape[1]) 
                
            loss_te[k], w = learning_by_penalized_gradient(y_tr, x_tr, w, param['gamma'],                                                                                                       param['lambda_'])
            y_pred = predict_labels_log(w, x_te)
            score.append(np.sum(y_pred == y_te) / len(y_te))
        else:
            w, _ = ridge_regression(y_tr, poly_tr, param['lambda_'])
            y_pred = predict_labels(w, poly_te)
            score.append(accuracy(y_pred, y_te))
                
        
    return np.mean(score), np.mean(loss_te)



def cross_validation_log(y, x, k_indices, gamma, lambda_):
    """Cross-validation for regularized logistic regression."""
    
    kfolds = k_indices.shape[0]
    accuracy = np.zeros(kfolds)
    loss_te = np.zeros(kfolds)
    if len(x.shape) == 1:
        w = 0
    else:
        w = np.zeros(x.shape[1]) 
        
    # k-fold    
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
        # accuracy
        y_pred = predict_labels_log(w, xte)
        accuracy[k] = np.sum(y_pred == yte) / len(yte)  
        
    return np.mean(accuracy), np.mean(loss_te)

#Trial single function
def find_best_parameters_general(labels, data, k_fold, seed, **param): 
    """Find the best parameters for ridge regression based on cross validation."""
    
    #Initialize
    k_idx = build_k_indices(labels, k_fold, seed)
    loss_te = np.ones((len(param['degrees']), len(param['lambdas'])))
    scores = np.ones((len(param['degrees']), len(param['lambdas'])))
    
    #Iterate over parameters
    for degree_idx, degree in enumerate(param['degrees']):
        for lambda_idx, lambda_ in enumerate(param['lambdas']):
            scores[degree_idx, lambda_idx], loss_te[degree_idx, lambda_idx]=                                                                        cross_validation_general(labels, data, k_idx, k_fold, 
                                                                    lambda_=lambda_, degree=degree)
            if 'gamma' in param:
                scores[degree_idx, lambda_idx], loss_te [degree_idx, lambda_idx]=                                                                    cross_validation_general(labels, data, k_idx, k_fold, 
                                                                    lambda_=lambda_, degree=degree, gamma=param['gamma'])
    #Select best parameters        
    best_HP_idx = np.unravel_index(np.argmax(scores), np.shape(scores))
    best_degree = param['degrees'][best_HP_idx[0]]
    best_lambda = param['lambdas'][best_HP_idx[1]]
    
    #Best score
    best_score = scores[best_HP_idx[0], best_HP_idx[1]]
    best_loss = loss_te[best_HP_idx[0], best_HP_idx[1]]

    
    return best_degree, best_lambda, best_score, scores, best_loss

def find_best_parameters(labels, data, k_fold, lambdas, degrees, seed): #rename with ridge????
    """Find the best parameters for ridge regression based on cross validation."""
    
    #Initialize
    k_idx = build_k_indices(labels, k_fold, seed)
    loss_te = np.ones((len(degrees), len(lambdas)))
    scores = np.ones((len(degrees), len(lambdas)))
    
    #Iterate over parameters
    for degree_idx, degree in enumerate(degrees):
        for lambda_idx, lambda_ in enumerate(lambdas):
            scores[degree_idx, lambda_idx] = cross_validation(labels, data, k_idx, k_fold, lambda_, degree)
            
    #Select best parameters        
    best_HP_idx = np.unravel_index(np.argmax(scores), np.shape(scores))
    best_degree = degrees[best_HP_idx[0]]
    best_lambda = lambdas[best_HP_idx[1]]
    
    #Best score
    best_score = score[best_HP_idx[0], best_HP_idx[1]]
    
    return best_degree, best_lambda, best_score, scores


def logistic_find_best_parameters(y, tx, lambdas, gamma, degrees):
    """Find the best parameters for logistic regression based on cross validation.""" 
    
    #Initialize
    loss_tr = np.zeros((len(lambdas), len(degrees)))
    loss_te = np.zeros((len(lambdas), len(degrees)))
    score = np.zeros((len(lambdas), len(degrees)))
    
    #Iterate over paramaters
    for lambda_idx, lambda_ in enumerate(lambdas):
        for degree_idx, degree in enumerate(degrees):          
            poly_x = build_poly(tx, degree)
            k_indices = build_k_indices(y, k_fold, seed)
            mean_score, mean_loss_te = cross_validation_log(y, poly_x, k_indices, lambda_, gamma)        
            score[lambda_idx, degree_idx] = mean_score
            loss_te[lambda_idx, degree_idx] = mean_loss_te
            
    ratio = score/loss_te
    
    #Select best parameters
    best_HP_idx = np.unravel_index(np.argmax(ratio), np.shape(ratio))
    best_degree = degrees[best_HP_idx[1]]
    best_lambda = lambdas[best_HP_idx[0]]
    
    #Best score & loss
    best_score = score[best_HP_idx[0], best_HP_idx[1]]
    best_loss = loss_te[best_HP_idx[0], best_HP_idx[1]]
    
    return best_lambda, best_degree, best_score, best_loss #change order

def make_predictions(estimated_data, labels, estimated_data_te, best_lambda, best_degree):
    """Use the best parameters to make the prediction for ridge regression."""
    
    #Build polynoms
    poly_data = build_poly(estimated_data, best_degree)
    poly_data_te = build_poly(estimated_data_te, best_degree)
    
    #Train
    weights, loss = ridge_regression(labels, poly_data, best_lambda)
    
    #Predict
    y_pred = predict_labels(weights, poly_data_te)
    
    return y_pred

def make_predictions_log(estimated_data, labels, estimated_data_te, best_lambda, best_degree, max_iters, gamma):
    """Use the best parameters to make the prediction for regularized logistic regression."""
    
    #Build polynoms
    poly_data = build_poly(estimated_data, best_degree)
    poly_data_te = build_poly(estimated_data_te, best_degree)
    
    #Train
    initial_w = np.zeros(poly_data.shape[1])
    weights, loss = reg_logistic_regression(labels, poly_data, best_lambda, initial_w, max_iters, gamma)
    
    #Predict
    y_pred = final_predict_labels_log(weights, poly_data_te)
    
    return y_pred


''''***********************************************************
********************** Data processing ************************
************************************************************'''

def divide_data(labels, raw_data):
    """Divide the data according to the number of jets (feature #22)."""
    
    # information of jets 
    jet0_data = raw_data[np.where(raw_data[:,22]== 0)[0],:]
    jet0_labels = labels[np.where(raw_data[:,22]== 0)[0]]
    jet1_data = raw_data[np.where(raw_data[:,22]== 1)[0],:]
    jet1_labels = labels[np.where(raw_data[:,22]== 1)[0]]
    jet2_data = raw_data[np.where(raw_data[:,22] > 1)[0],:]
    jet2_labels = labels[np.where(raw_data[:,22] > 1)[0]]
    
    return jet0_labels, jet0_data, jet1_labels, jet1_data, jet2_labels, jet2_data


def find_null_variance_features(data):
    """Find for each jet features with a variance equal to 0."""
    
    variance = np.var(data, axis = 0)
    no_var = np.where(variance == 0)
    print('Columns with variance = 0 :', no_var) #PRINT TO MOVE TO RUN.PY
    
    return no_var


def remove_novar_features(data, data_te):
    """Remove features with a variance equal to 0."""
    
    no_var_columns = find_null_variance_features(data)
    clean_data = np.delete(data, no_var_columns, axis = 1)
    clean_data_te = np.delete(data_te, no_var_columns, axis = 1)
    print('New data shape :', clean_data.shape) #PRINT TO MOVE TO RUN.PY
    
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
    """Standardize the test data along the feature axis with known means and standard deviations."""
    
    test_data = meaningless_to_nan(test_data)
    standardized_data_te = (test_data - mean_train) / std_train
    
    return standardized_data_te


def column_estimation_train(data):
    """Estimate the NaN in column 0 based on the other features using least-squares regression."""
    
    #Define parameters
    chosen_feature = 0
    degree = 1
    submatrix = np.delete(data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    
    #Add to samples to be ignored
    for sample in range(n_samples):
        if (np.isnan(data[sample,chosen_feature])):
            samples.append(sample)
            
    print(len(samples), 'NaN lines found') #PRINT TO MOVE TO RUN.PY
    
    #Extract data, build poly, train and predict values
    submatrix_bis = np.delete(submatrix, samples, axis = 0) 
    labels_bis = np.delete(data[:,chosen_feature], samples, axis = 0)
    poly = build_poly(submatrix_bis, degree)
    weights_train, _ = least_squares(labels_bis, poly)
    poly_te = build_poly(submatrix[samples,:], degree)
    x_pred = np.dot(poly_te, weights_train)
    data[samples, chosen_feature] = x_pred
    
    return data, weights_train

def column_estimation_test(data, weights_train):
    """"Estimate the NaN in column 0 based on the other features AND the train weights."""
    
    #Define parameters
    chosen_feature = 0
    degree = 1
    submatrix = np.delete(data,chosen_feature, axis = 1)
    n_samples, n_features = np.shape(submatrix)
    samples = []
    
    #Add to samples to be ignored
    for sample in range(n_samples):
        if (np.isnan(data[sample,chosen_feature])):
            samples.append(sample)
            
    #Extract data, build poly, train and predict values        
    submatrix_bis = np.delete(submatrix, samples, axis = 0)
    labels_bis = np.delete(data[:,chosen_feature], samples, axis = 0)
    poly_te = build_poly(submatrix[samples,:], degree)
    x_pred = np.dot(poly_te, weights_train)
    data[samples, chosen_feature] = x_pred
    
    return data
