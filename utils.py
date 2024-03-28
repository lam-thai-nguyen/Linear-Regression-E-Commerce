import numpy as np
import matplotlib.pyplot as plt


def plot_cost_iter(cost_record: list[float]) -> None:
    """
    Plot the relationship between cost value and number of iterations
    """
    plt.plot(cost_record)
    plt.xlabel("Iter")
    plt.ylabel("Cost")
    plt.show()
    
    
def plot_model(features: np.array, targets: np.array, w: float, b: float) -> None:
    """
    Plot datapoints and LR model
    """
    plt.scatter(features, targets, marker='x', c='b')
    plt.plot(features, w * features + b, c='r')
    plt.show()
    
    
def z_normalize(X: np.array):
    """
    After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.
    
    Argument: X: features
    Returns: normalized_X
    """
    # Feature mean
    mu = np.mean(X, axis=0)
    # Feature standard deviation
    std = np.std(X, axis=0)
    
    X_norm = (X - mu) / std
    
    return X_norm
    