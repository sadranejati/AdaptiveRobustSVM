from models import *

if __name__ == "__main__":
    df = generate_2d_classification_data(n_samples=500, mislabel_percentage=0.1, cluster_distance=5.0, random_state=42)
    plot_2d_classification_data(df)
