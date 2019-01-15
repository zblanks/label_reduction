import numpy as np
from core.coord_desc import run_coord_desc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _build_v_mat(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the V matrix from X and y
    """
    # Define a placeholder for V
    nclasses = len(np.unique(y))
    p = X.shape[1]
    V = np.empty(shape=(nclasses, p))

    # To compute the mean value for a given class, we need to get the
    # indices that belong to it
    idx_list = [np.where(y == i)[0] for i in range(nclasses)]

    # Using these indices, we will now compute the mean point for the
    # label
    for i in range(nclasses):
        V[i, :] = X[idx_list[i], :].mean(axis=0)

    return V


def _compute_var(X: np.ndarray, y: np.ndarray, label: int) -> float:
    """
    Computes the variance of a particular label
    """

    # First identify the samples that belong to `label`
    idx = np.where(y == label)[0]

    # Using this, subset the matrix, `X`, and then compute the variance
    # using NumPy functions
    return X[idx, :].var()


def _kmeans_mean(V: np.ndarray, k: int, ninit: int):
    """
    Runs the kmeans-means variant of label reduction
    """

    # Cluster the centroids
    kmeans = KMeans(n_clusters=k, random_state=17, n_jobs=-1, n_init=ninit)
    return kmeans.fit_predict(V)


def group_labels(X: np.ndarray, y: np.ndarray, k: int,
                 group_algo: str, ninit=10) -> list:
    """
    Helper function to group the labels in the data set given a grouping
    algorithm
    """

    # We need to standardize the training data before grouping it
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # All of our approaches work with the mean matrix, V, so we need to
    # compute this
    V = _build_v_mat(X, y)

    # The coordinate descent algorithm requires the variances of each of
    # the labels also be computed
    uniq_labels = np.unique(y)
    label_vars = np.array([_compute_var(X, y, i) for i in uniq_labels])

    # Run the label grouping algorithm
    if group_algo == "kmm":
        # The kmeans algorithm automatically runs in parallel so we don't
        # need to call it using joblib
        label_groups = _kmeans_mean(V, k, ninit)
    else:
        label_groups = run_coord_desc(V, label_vars, y, k, ninit)

    return label_groups
