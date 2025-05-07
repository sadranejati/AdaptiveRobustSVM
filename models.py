import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

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


def plot_2d_classification_data(df):
    """
    Visualize a 2D classification dataset.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'x1', 'x2', and 'label' (-1 or 1).
    """
    plt.figure(figsize=(8, 6))
    for label in [-1, 1]:
        subset = df[df['label'] == label]
        plt.scatter(subset['x1'], subset['x2'], label=f'Class {label}', alpha=0.6)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D Classification Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()
