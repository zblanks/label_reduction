import numpy as np
from core.label_group_milp import lp_heuristic
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from itertools import product
import networkx as nx
import community


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


def _kmeans_mean(V: np.ndarray, k: int, rng: np.random.RandomState,
                 ninit: int):
    """
    Runs the kmeans-means variant of label reduction
    """

    # Cluster the centroids
    kmeans = KMeans(n_clusters=k, random_state=rng, n_jobs=-1, n_init=ninit)
    return kmeans.fit_predict(V)


def _community_detection(V: np.ndarray, metric: str,
                         rng: np.random.RandomState):
    """
    Implements the community detection approach to grouping labels
    """

    # Implement the community detection approach with different similarity
    # or distance metrics (our options are the RBF kernel, L2 distance,
    # Earth Mover's Distance, and the L-infinity distance

    # We're working directly with a similarity metric
    if 'rbf' in metric:
        S = rbf_kernel(V)

    # Otherwise we're working with a distance metric and need to
    # treat it differently
    else:
        if 'l2' in metric:
            D = pairwise_distances(V, metric='euclidean', n_jobs=-1)
        elif 'emd' in metric:
            # Sklearn doesn't have a built-in function for the
            # EMD so we have to implement the parallel code ourselves
            n = V.shape[0]
            idx = product(range(n), range(n))
            f = delayed(wasserstein_distance)
            res = Parallel(n_jobs=-1, verbose=0)(
                f(V[val[0], :], V[val[1], :]) for val in idx
            )

            # Get the array
            D = np.array(res).reshape(n, n)
        else:
            D = pairwise_distances(V, metric='chebyshev', n_jobs=-1)

        # Convert the distance matrix into a similarity matrix
        S = 1 / np.exp(D)

    # Infer the communities using the Louvain algorithm
    partition = community.best_partition(nx.Graph(S), random_state=rng)

    # Convert the output to an array so it's in the same format as other
    # return results
    return np.array([partition[val] for val in partition.keys()])


def group_labels(X: np.ndarray, y: np.ndarray, k: int,
                 group_algo: str, rng: np.random.RandomState,
                 ninit=10) -> np.ndarray:
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

    # Run the label grouping algorithm
    if group_algo == "kmm":
        # The kmeans algorithm automatically runs in parallel so we don't
        # need to call it using joblib
        label_groups = _kmeans_mean(V, k, rng, ninit)

    elif 'comm' in group_algo:
        label_groups = _community_detection(V, group_algo, rng)

    else:
        label_groups = lp_heuristic(V, k)

    return label_groups
