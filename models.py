import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import time 
from sklearn.datasets import make_blobs
gp.setParam('OutputFlag', 0)

def generate_2d_classification_data(n_samples=1000, mislabel_percentage=0.1, 
                                     cluster_distance=4.0, random_state=None):
    """
    Generate a 2D binary classification dataset with two clusters and optional mislabeling.
    
    Parameters:
    - n_samples (int): Total number of samples.
    - mislabel_percentage (float): Fraction of labels to randomly flip.
    - cluster_distance (float): Distance between the centers of the two clusters.
    - random_state (int or None): Seed for reproducibility.

    Returns:
    - pd.DataFrame: DataFrame with columns 'x1', 'x2', and 'label' (-1 or 1).
    """
    rng = np.random.default_rng(seed=random_state)

    # Define two cluster centers based on the distance
    centers = np.array([[0, 0], [cluster_distance, cluster_distance]])

    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2,
                      cluster_std=1.0, random_state=random_state)

    # Convert labels from {0, 1} to {-1, 1}
    y = 2 * y - 1

    # Introduce mislabeling
    if mislabel_percentage > 0:
        n_mislabeled = int(n_samples * mislabel_percentage)
        indices_to_flip = rng.choice(n_samples, size=n_mislabeled, replace=False)
        y[indices_to_flip] *= -1  # Flip -1 to 1 and vice versa

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y

    return df


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def plot_2d_classification_data(df, w=None, b=None, show_hinge_loss=False, delta=None):
    """
    Visualize a 2D classification dataset, and optionally plot a hyperplane defined by w and b.
    Also annotate each point with its w^T x - b value.
    If delta is provided, circle the points that are mislabeled.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'x1', 'x2', and 'label' (-1 or 1).
    - w (np.array or list or None): Weight vector (length 2).
    - b (float or None): Bias term.
    - show_hinge_loss (bool): Whether to annotate points with w^T x - b values.
    - delta (np.array or list or None): 1D array indicating mislabeled points (1 = mislabeled).
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the points
    for label in [-1, 1]:
        subset = df[df['label'] == label]
        plt.scatter(subset['x1'], subset['x2'], label=f'Class {label}', alpha=0.6)
    
    # If w and b are given, plot the hyperplane
    if w is not None and b is not None:
        w = np.array(w)  # Make sure it's numpy array
        
        x1_min, x1_max = df['x1'].min() - 1, df['x1'].max() + 1
        
        if w[1] != 0:
            x1_vals = np.linspace(x1_min, x1_max, 200)
            x2_vals = (b - w[0] * x1_vals) / w[1]
            plt.plot(x1_vals, x2_vals, 'k-', label='Hyperplane')
        else:
            x_val = b / w[0]
            plt.axvline(x=x_val, color='k', linestyle='-', label='Hyperplane')
        
        # Now calculate and annotate w^T x - b for each point
        if show_hinge_loss:
            X = df[['x1', 'x2']].values
            wx_minus_b = X @ w - b
            
            for i, (x, y_point) in enumerate(zip(X, wx_minus_b)):
                text = f"${{w^Tx-b}}$={y_point:.2f}"
                plt.annotate(
                    text,
                    (x[0], x[1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha='left',
                    fontsize=8,
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
                )

    # If delta is provided, circle the mislabeled points
    if delta is not None:
        X = df[['x1', 'x2']].values
        mislabeled_indices = np.where(delta == 1)[0]
        
        for idx in mislabeled_indices:
            x, y_ = X[idx]
            circle = plt.Circle((x, y_), radius=0.5, color='red', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D Classification Dataset with w^T x - b Values and Mislabeled Points")
    plt.legend()
    plt.grid(True)
    plt.show()
    



def label_enhanced(X, y, Gamma=0, M=1000):
    n, d = len(y), len(X[0])  # Number of data points and features
    Gamma = Gamma * n
    cpu_time = time.time()
    # Create model
    model = gp.Model("Mixed-Integer Optimization")
    
    # Decision variables
    xi = model.addMVar(n, lb=0, vtype=GRB.CONTINUOUS, name="xi")
    ri = model.addMVar(n, lb=0, vtype=GRB.CONTINUOUS, name="ri")
    q = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="q")
    s = model.addMVar(n, vtype=GRB.BINARY, name="s")
    w = model.addMVar(d, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
    b = model.addVar(lb=-GRB.INFINITY ,vtype=GRB.CONTINUOUS, name="b")
    
    # Objective function
    model.setObjective(gp.quicksum(xi) + Gamma * q + gp.quicksum(ri), GRB.MINIMIZE)
    model.addConstr(q + ri + xi >= 0, name="c1")
    model.addConstr(q + ri + xi >= 1 + y * ((X @ w) - b), name="c2")
    model.addConstr(xi >= 1 - y * ((X @ w) - b), name="c3")
    model.addConstr(xi <= 1 - y * ((X @ w) - b) + M * (1 - s), name="c4")
    model.addConstr(xi <= M * s, name="c5")

    
    # Optimize
    model.optimize()
    
    # Retrieve results
    if model.status == GRB.OPTIMAL:
        results = {
            "solver": model.Runtime,
            'cpu':time.time() - cpu_time,
            "w": [w[j].x for j in range(d)],
            "b": b.x,
        }
        return results
    else:
        return None
    

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def maximize_sum_hinge_correct(X, y, w, b, Gamma, M=1000):
    n_samples = X.shape[0]

    model = gp.Model()

    # Variables
    delta_y = model.addVars(n_samples, lb=0.0, ub=1.0, name="delta_y")
    xi = model.addVars(n_samples, lb=0.0, name="xi")
    z = model.addVars(n_samples, vtype=GRB.BINARY, name="z")

    # Objective: maximize sum of xi
    model.setObjective(gp.quicksum(xi[i] for i in range(n_samples)), GRB.MAXIMIZE)

    # Constraints
    for i in range(n_samples):
        margin = w @ X[i] - b
        expr = 1 - y[i] * (1 - 2 * delta_y[i]) * margin

        model.addConstr(xi[i] >= expr)
        model.addConstr(xi[i] >= 0)
        model.addConstr(xi[i] <= M * z[i])
        model.addConstr(xi[i] <= expr + M * (1 - z[i]))

    model.addConstr(gp.quicksum(delta_y)<= Gamma*n_samples, name="budget")
    # Optimize
    model.setParam('OutputFlag', 0)
    model.optimize()

    # Extract results
    delta_y_opt = np.array([delta_y[i].X for i in range(n_samples)])
    xi_values = np.array([xi[i].X for i in range(n_samples)])
    obj_val = model.objVal

    return delta_y_opt, xi_values, obj_val
