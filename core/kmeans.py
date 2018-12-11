from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def kmeans_all(X, y, k: int):
    """
    Runs the k-means all variant of label reduction
    """

    # Run the k-means clustering algorithm
    kmeans = KMeans(n_clusters=k, random_state=17, n_jobs=-1, verbose=True)
    assignments = kmeans.fit_predict(X)

    # Map the labels to the new meta-classes
    nclass = len(np.unique(y))
    idx = [np.where(y == i) for i in range(nclass)]
    metaclasses = np.array(
        [np.bincount(assignments[idx[i]]).argmax() for i in range(nclass)]
    )

    label_groups = [np.where(metaclasses == i)[0].tolist() for i in
                    range(k)]
    return label_groups


def kmeans_mean(X, y, k: int):
    """
    Runs the kmeans-means variant of label reduction
    """

    # Standardize the data before clustering
    scaler = StandardScaler()
    X_new = np.copy(X)
    X_new = scaler.fit_transform(X_new)

    # Get the centroid representation for each original class
    nclass = len(np.unique(y))
    idx = [np.where(y == i)[0] for i in range(nclass)]
    centroids = [np.empty(shape=(1, X_new.shape[1]))] * nclass
    for i in range(nclass):
        centroids[i] = X_new[idx[i], :].mean(axis=0).reshape(1, -1)
    centroids = np.concatenate(centroids, axis=0)

    # Cluster the centroids
    kmeans = KMeans(n_clusters=k, random_state=17, n_jobs=-1)
    assignments = kmeans.fit_predict(centroids)

    # Get the assignments into the expected format
    label_groups = [np.where(assignments == i)[0].tolist() for i in
                    range(k)]
    return label_groups
