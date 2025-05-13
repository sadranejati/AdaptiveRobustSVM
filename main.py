from models import *

if __name__ == "__main__":
    df = generate_2d_classification_data(n_samples=50, mislabel_percentage=0.1, cluster_distance=5.0, random_state=42)
    

 

    #df = pd.DataFrame(X, columns=["x1", "x2"])
    #df["label"] = y

    
    results = label_enhanced(df[['x1', 'x2']].to_numpy(), df['label'].to_numpy(), Gamma=0.1, M=1000000)
    print(results)
 

    delta, _, _ = (maximize_sum_hinge_correct(df[['x1', 'x2']].to_numpy(), df['label'].to_numpy(),  results["w"], results["b"],Gamma=0.1, M=1000000))
    plot_2d_classification_data(df, results["w"], results["b"], delta=delta)
