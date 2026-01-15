import pandas as pd
import numpy as np
from clustering import split_frequent_drivers, _scale_cluster_features, assign_clusters_kmeans, assign_clusters_gmm

def test_split_frequent_drivers():
    print("Testing split_frequent_drivers...")
    data = {
        "plate": ["A", "B", "C", "D"],
        "total_passes": [10, 25, 5, 50],
    }
    df = pd.DataFrame(data)
    
    frequent, rare = split_frequent_drivers(df, min_total_passes=20)
    
    assert len(frequent) == 2, f"Expected 2 frequent, got {len(frequent)}"
    assert sorted(frequent["plate"].tolist()) == ["B", "D"]
    assert len(rare) == 2, f"Expected 2 rare, got {len(rare)}"
    assert sorted(rare["plate"].tolist()) == ["A", "C"]
    print("PASS")

def test_scale_cluster_features_fit_on_subset():
    print("Testing scale_cluster_features...")
    df_train = pd.DataFrame({"val": [9.0, 10.0, 11.0]})
    df_all = pd.DataFrame({"val": [9.0, 10.0, 11.0, 100.0]})
    
    cols = ["val"]
    
    X_subset_fit, scaler = _scale_cluster_features(df_all, cols, train_df=df_train)
    
    assert abs(X_subset_fit[1][0]) < 0.1, "Mean of train set should be approx 0"
    assert X_subset_fit[3][0] > 10.0, "Outlier should be far away"
    print("PASS")

def test_assign_clusters_kmeans():
    print("Testing assign_clusters_kmeans...")
    # 3 clusters around 0, 10, 20
    freq_data = {
        "val": [0, 0.1, -0.1, 10, 10.1, 9.9, 20, 20.1, 19.9],
        "plate": [f"F{i}" for i in range(9)]
    }
    freq_df = pd.DataFrame(freq_data)
    
    # Rare: one close to 0, one far away (50)
    rare_data = {
        "val": [0.05, 50.0],
        "plate": ["R1", "R2"]
    }
    rare_df = pd.DataFrame(rare_data)
    
    full_df, model, threshold = assign_clusters_kmeans(
        freq_df, rare_df, ["val"], k=3, confidence_threshold_percentile=99
    )
    
    # R1 should be assigned (probably cluster 0, 1 or 2, but not -1)
    r1 = full_df[full_df["plate"] == "R1"].iloc[0]
    assert r1["cluster_label"] != -1, "R1 should be assigned"
    
    # R2 should be unknown (-1)
    r2 = full_df[full_df["plate"] == "R2"].iloc[0]
    assert r2["cluster_label"] == -1, f"R2 should be -1, got {r2['cluster_label']}"
    assert r2["is_rare"] == True
    print("PASS")

if __name__ == "__main__":
    try:
        test_split_frequent_drivers()
        test_scale_cluster_features_fit_on_subset()
        test_assign_clusters_kmeans()
        print("\nALL PREILIMINARY CHECKS PASSED")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
