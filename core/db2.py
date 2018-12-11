from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


def prepare_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Prepares the data by computing the class centroids
    """

    # Standardize the data before grouping
    scaler = StandardScaler()
    X_new = np.copy(X)
    X_new = scaler.fit_transform(X_new)

    # Get the centroids for each label
    unique_labs = np.unique(y).astype(int)
    nclasses = len(unique_labs)
    idx = [np.where(y == i)[0] for i in range(nclasses)]
    centroids = [np.empty(shape=(1, X_new.shape[1]))] * nclasses
    for i in range(nclasses):
        centroids[i] = X_new[idx[i], :].mean(axis=0).reshape(1, -1)
    centroids = np.concatenate(centroids, axis=0)
    return centroids, unique_labs


def divide_by_two(X: np.ndarray, y: np.ndarray, tree_dict: dict) -> dict:
    """
    Performs the DB2 hierarchy formation technique
    """

    # Base case: len(y) == 1 --> do nothing
    if len(y) == 1:
        return tree_dict
    # Base case: len(y) == 2 --> the children are just the values of the parent
    elif len(y) == 2:
        tree_dict[str(y.tolist())] = [[y[0]], [y[1]]]
        return tree_dict
    # Recursive case: cluster the values and call again
    else:
        # Cluster the data
        kmeans = KMeans(n_clusters=2, random_state=17)
        clusters = kmeans.fit_predict(X[y, :])

        # Get the cluster predictions for the first and second clusters
        zero_labs = y[np.where(clusters == 0)]
        one_labs = y[np.where(clusters == 1)]

        # Update the tree_dict
        tree_dict[str(y.tolist())] = [zero_labs.tolist(), one_labs.tolist()]

        # Recursively call the DB2 method again for both the zero and one
        # labels
        tree_dict = divide_by_two(X, zero_labs, tree_dict)
        tree_dict = divide_by_two(X, one_labs, tree_dict)
        return tree_dict


def run_db2(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Runs the DB2 algorithm
    """

    # Prepare the data
    centroids, labels = prepare_data(X, y)

    # Get the DB2 hierarchy
    return divide_by_two(centroids, labels, {})
