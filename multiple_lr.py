import numpy as np
import matplotlib.pyplot as plt
from utils import plot_cost_iter


def _compute_cost(X: np.array, y: np.array, w: np.array, b: float) -> float:
    """
    Returns the total cost for multiple LR
    """
    m = X.shape[0]
    cost = 0.0
    
    for i in range(m):
        y_hat = np.dot(w, X[i]) + b
        e = (y_hat - y[i]) ** 2
        cost += e
        
    total_cost = cost / (2 * m)
    
    return total_cost


def _compute_gradient(X: np.array, y: np.array, w: np.array, b: float) -> list[list[float], float]:
    """
    Compute the derivative of cost wrt. w_i and b

    Returns: dJ_dw and dJ_db
    """
    m, n = X.shape
    dJ_dw = np.zeros(n)
    dJ_db = 0.0
    
    for i in range(m):
        y_hat = np.dot(w, X[i]) + b
        e = y_hat - y[i]
        dJ_db += e
        
        for j in range(n):
            dJ_dw[j] += e * X[i, j]
        
    dJ_db /= m
    dJ_dw /= m
    
    return dJ_dw, dJ_db


def gradient_descent(X: np.array, y: np.array, w_init: list[float], b_init: float, num_iters: int, learning_rate: float, plot_cost_per_iter: bool) -> list[list[float], float]:
    """
    Compute optimal values for w and b
    
    Returns: w_optimal [list[float]] and b_optimal [float]
    """
    w, b = w_init, b_init
    alpha = learning_rate
    cost_record = []
    
    for i in range(num_iters):
        dJ_dw, dJ_db = _compute_gradient(X, y, w, b)
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db
        cost = _compute_cost(X, y, w, b)
        cost_record.append(cost)
        
        if i % (num_iters // 10) == 0:
            print(f"{i}th cost: {cost}")
            
    if plot_cost_per_iter:
        plot_cost_iter(cost_record)
            
    return w, b
