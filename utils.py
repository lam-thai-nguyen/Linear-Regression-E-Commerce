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
    