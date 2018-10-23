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

def column_estimation_train(nan_data):
    
    n_samples, n_features = np.shape(nan_data)
    nan_columns = nan_find_columns(nan_data)
    submatrix = np.delete(nan_data, nan_columns, axis = 1)
    weights_train = np.zeros((len(nan_columns),n_features-len(nan_columns)))

    for f_ind, chosen_feature in enumerate(nan_columns):
        samples = []
        for sample in range(n_samples):
            if np.isnan(nan_data[sample,chosen_feature]):
                samples.append(sample)
        nan_lines = np.unique(samples)
        submatrix0 = np.delete(submatrix, nan_lines, axis = 0)
        labels0 = np.delete(nan_data[:,chosen_feature], nan_lines, axis = 0)
        
        weights_train[f_ind,:], _ = least_squares(labels0, submatrix0)
        x_pred = np.dot(submatrix[nan_lines,:], weights_train[f_ind,:])
        nan_data[nan_lines, chosen_feature] = x_pred
        
    return nan_data,nan_columns,weights_train

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
        
        # logistic regression
        #initial_w = np.zeros((len(poly_tr[0]),1))
        #w,loss_tr = logistic_regression(y_tr,poly_tr, initial_w,max_iters,lambda_)
        
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

## LOGISTIC REGRESSION

def sigmoid(t):
    """Apply stable sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))
    
def calculate_loss(y, tx, w):
    """Compute the cost by negative log-likelihood."""
    data = tx.dot(w)
    data_std,_,_ = standardize_train(data)
    pred = sigmoid(data_std)
    loss = y.T.dot(np.log(pred)) + (1.0 - y).T.dot(np.log(1.0 - pred))
    return np.squeeze(- loss) # why squeeze?

def calculate_gradient(y, tx, w):
    """Compute the gradient of loss for sigmoidal prediction."""
    pred = sigmoid(tx.dot(w))
    err = pred - y
    grad = np.transpose(tx) @ err
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    new_w = w - gamma * grad
    return loss, new_w


def logistic_regression(y, x, initial_w,max_iters, gamma):
    """
    Does one step of gradient descent using logistic regression. 
    Return the loss and the updated weight w.
    """
    threshold = 1e-09
    losses = []
    w = initial_w
    # build tx including w_0 weight
    #x = np.c_[np.ones((y.shape[0], 1)), x]
    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, x, w, gamma)
        # log info
        #if i % 10 == 0:
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

def learning_by_gradient_descent_reg(y, tx, w, gamma,lambda_):
    """
    Do one step of gradient descent using regularized logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    new_w = w - gamma * grad - gamma * lambda_*w
    return loss, new_w

def reg_logistic_regression(y, x, lambda_, initial_w,max_iters, gamma):
    """
    Do one step of gradient descent using reg_logistic regression.
    Return the loss and the updated w.
    """
    
    threshold = 1e-8
    losses = []
    w = initial_w
    #x = np.c_[np.ones((y.shape[0], 1)), x]
    # start the logistic regression
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent_reg(y, x, w, gamma,lambda_)
        # log info
        #if i % 10 == 0:
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
