import h5py
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def run_benchmark(data: h5py.Dataset, path: str, n_cluster: int,
                  n_init=10) -> None:
    """Runs the k-means benchmark
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data[:, :-1])
    kmeans = MiniBatchKMeans(n_clusters=n_cluster, verbose=True,
                             random_state=17, n_init=n_init)
    clusters = kmeans.fit_predict(data)

    # Save the clusters to disk
    with h5py.File(path, "w") as f:
        f.create_dataset("cluster_labels", data=clusters)
    return None
