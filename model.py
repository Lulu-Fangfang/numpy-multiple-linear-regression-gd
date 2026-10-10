"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.RandomState(seed)

    indices = rng.permutation(X.shape[0])
    return  X[indices] , y[indices]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n, d = np.shape(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    X_tr = X[:n_train,:]
    X_val = X[n_train:(n_train+n_val), :]
    X_test = X[(n_train + n_val): (n_train+n_val+n_test), :]

    y_tr = y[:n_train]
    y_val = y[n_train:(n_train+n_val)]
    y_test = y[(n_train + n_val):(n_train + n_val + n_test)]

    return (X_tr, y_tr,X_val,y_val,X_test,y_test)

# Step 3 - compute_feature_stats
import numpy as np


def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    X  = np.asarray(X, dtype = np.float64)
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std[std == 0] = 1
    return mean,std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    X = np.asarray(X, dtype = np.float64)
    mean = np.asarray(mean, dtype = np.float64)
    std = np.asarray(std, dtype = np.float64)

    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    X = np.asarray(X)

    ones = np.ones((X.shape[0], 1), dtype = X.dtype)
    return np.concatenate((ones, X), axis = 1)

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X = np.asarray(X, dtype = np.float64)

    X_scaled = (X - mean) / std

    ones = np.ones((X.shape[0], 1))

    return add_bias_column(X_scaled)

# Step 7 - predict_linear (not yet solved)
# TODO: implement

# Step 8 - mse_loss (not yet solved)
# TODO: implement

# Step 9 - mse_gradient (not yet solved)
# TODO: implement

# Step 10 - normal_equation (not yet solved)
# TODO: implement

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

