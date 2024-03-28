import math
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_cost_iter


def _compute_cost(x: np.array, y: np.array, w: float, b: float) -> float:
    """
    Returns total cost for given pair of y-hat and y-true
    """
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        y_hat = w * x[i] + b
        e = (y_hat - y[i]) ** 2
        cost += e

    total_cost = cost / (2 * m)

    return total_cost


def _compute_gradient(x: np.array, y: np.array, w: float, b: float) -> list[float]:
    """
    Compute the derivative of cost wrt. w and b

    Returns: dJ_dw and dJ_db
    """
    m = x.shape[0]
    dJ_dw, dJ_db = 0.0, 0.0

    for i in range(m):
        y_hat = w * x[i] + b
        e = y_hat - y[i]
        dJ_dw += e * x[i]
        dJ_db += e

    dJ_dw /= m
    dJ_db /= m

    return dJ_dw, dJ_db


def gradient_descent(x: np.array, y: np.array, w_init: float, b_init: float, num_iters: int, learning_rate: float, plot_cost_per_iter: bool) -> list[float]:
    """
    Perform gradient descent

    Returns: optimal w and b
    """
    w, b = w_init, b_init
    alpha = learning_rate
    cost_record = []
    
    for i in range(num_iters):
        dJ_dw, dJ_db = _compute_gradient(x, y, w, b)
        w = w - alpha * dJ_dw
        b = b - alpha * dJ_db
        cost = _compute_cost(x, y, w, b)
        cost_record.append(cost)
        
        if i % math.ceil(num_iters / 10) == 0:
            print(f"{i}th cost: {cost}")
            
    if plot_cost_per_iter:
        plot_cost_iter(cost_record)
            
    return w, b
