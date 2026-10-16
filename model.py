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

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    return np.sum((y_true-y_pred)**2)/ len(y_true)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    mse = X.T @ (y_pred - y_true)

    return  (2 * mse) / X.shape[0]

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(
        loc = 0.0,
        scale = 0.01,
        size = n_features
    )

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    n = X.shape[0]
    y_pred = X @ weights
    gradiant = ( 2 / n) * (X.T @ (y_pred - y))
    return weights - lr * gradiant

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    train_pred = X_train @ weights
    val_pred = X_val @ weights

    train_loss = float(np.mean((train_pred - y_train)**2))
    val_loss = float(np.mean((val_pred - y_val) ** 2))

    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = weights.copy()

        wait = 0
        stop = False
    else:
        wait += 1
        stop = wait >= patience
        
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    weights = initialize_weights(n_features, seed)

    return {
        "weights": weights,
        "best_weights": weights.copy(),
        "best_val_loss": np.inf,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped": False, 
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    state["weights"] = gd_step(
        X_train, y_train, state["weights"], lr
    )

    train_loss, val_loss = epoch_train_val_losses(
        X_train, y_train, X_val, y_val, state["weights"]
    )

    train_loss, val_loss = epoch_train_val_losses(
        X_train, y_train, X_val, y_val, state["weights"]
    )

    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)

    (
        state["best_val_loss"],
        state["wait"],
        state["best_weights"],
        state["stopped"],
    )= update_early_stop_state(
        val_loss,
        state["best_val_loss"],
        state["wait"],
        state["weights"],
        state["best_weights"],
        patience,
    )

    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    n_features = X_train.shape[1]

    state = init_training_state(n_features, seed)

    for _ in range(epochs):
        state = run_one_epoch(
            state, X_train, y_train, X_val,
            y_val,lr, patience
        )

        if state["stopped"]:
            break

    return (
        state["best_weights"],
        state["train_losses"],
        state["val_losses"]
    )

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true - y_pred)))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    return float(np.sqrt(np.mean((y_pred-y_true)**2)))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    if y_true.size < 2:
        return float("nan")
    # TODO: Compute the coefficient of determination R^2.
    residual_sum = np.sum((y_true - y_pred)**2)
    total_sum = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (residual_sum/total_sum)

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    mae = mean_absolute_error(y_true, y_pred)

    rmse = root_mean_squared_error(y_true, y_pred)

    r2 = r_squared(y_true, y_pred)

    return {
        'mae':mae, 
        'rmse':rmse , 
        'r2':r2,
        }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    # TODO: Return epoch indices and loss series for external plotting...
    epochs = list(range(1, len(train_losses) + 1))
    train = [float(loss) for loss in train_losses]
    val = [float(loss) for loss in val_losses]
    
    
    return epochs, train, val

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    # TODO: Compute the L2 distance between two weight vectors
    result = np.mean(np.sqrt(np.sum((w_gd - w_closed)**2)))
    return result

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    # TODO: Build the initial LinearRegressionGD-style model dictionary...
    
    
    
    return {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'seed': seed,
        'weights': None,
        'normal_weights': None,
        'mean': None,
        'std': None,
        'train_losses': [],
        'val_losses': []
    }

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    # TODO: Fit model with train stats, design matrices, GD, and normal eq
    X_train = np.asarray(X_train, dtype = float)
    X_val = np.asarray(X_val, dtype = float)
    y_train = np.asarray(y_train, dtype = float)
    y_val = np.asarray(y_val, dtype = float)

    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)

    std = np.where(std == 0, 1.0, std)
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std

    train_design = np.column_stack([
        np.ones(X_train.shape[0]),
        X_train_scaled,
    ])

    val_design = np.column_stack(
        [
            np.ones(X_val.shape[0]),
            X_val_scaled,
        ]
    )

    learning_rate = model["learning_rate"]
    epochs = model["epochs"]
    patience = model["patience"]

    rng = np.random.default_rng(model["seed"])
    weights = rng.normal(
        loc = 0.0,
        scale = 0.01,
        size = train_design.shape[1],
    )


    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_weights = weights.copy()
    no_improvement = 0

    for _ in range(epochs):
        error = train_design @ weights - y_train
        gradient = (
            2.0 / len(y_train)
        ) * (train_design.T @ error)

        weights = weights - learning_rate * gradient

        train_loss = float(
            np.mean((train_design @ weights - y_train) ** 2)
        )
        val_loss = float(
            np.mean((val_design @ weights - y_val) ** 2)
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = weights.copy()
            no_improvement = 0
        else:
            no_improvement += 1

            if patience is not None and no_improvement >= patience:
                break
    normal_weights = (
        np.linalg.pinv(train_design.T @ train_design)
        @ train_design.T
        @ y_train
    )

    model["mean"] = mean
    model["std"] = std
    model["weights"] = best_weights
    model["normal_weights"] = normal_weights
    model["train_losses"] = train_losses
    model["val_losses"] = val_losses

    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    # TODO: Return predicted targets for raw X using the fitted model.
    X = np.asarray(X, dtype = float)
    X_scaled = (X - model["mean"]) / model["std"]

    design = np.column_stack(
        [
            np.ones(X.shape[0]),
            X_scaled,
        ]
    )

    return design @ model["weights"]

# Step 27 - score_lr_model
def score_lr_model(model, X, y):
    # TODO: Predict on raw features and return MAE, RMSE, and R^2 metrics.
    y = np.asarray(y, dtype = float).reshape(-1)
    y_pred = predict_lr_model(model, X)

    # error = y - y_pred

    mae = mean_absolute_error(y, y_pred)

    rmse = root_mean_squared_error(y, y_pred)

    r2 = r_squared(y, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

