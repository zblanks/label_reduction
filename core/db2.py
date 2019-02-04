from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from time import time
from itertools import repeat
from core.hc import remap_labels, train_node
from multiprocessing import Pool


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


def db2_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
              tree_dict: dict, rng: np.random.RandomState, estimator: str,
              niter=10) -> dict:
    """
    Implements the DB2 benchmark
    """

    # Scale and reduce the dimensionality of the data only for the starting
    # data since they did not feed different features to each of the parent
    # nodes in the DB2 tree
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    pca = PCA(n_components=50, random_state=rng)
    X_train = pca.fit_transform(X_train)

    # Get the indices for each of the parents in the DB2 tree
    nkeys = len(tree_dict.keys())
    idx = [np.empty(shape=(y_train.shape[0],), dtype=bool)] * nkeys
    for (i, key) in enumerate(tree_dict.keys()):
        labels = tree_dict[key][0] + tree_dict[key][1]
        idx[i] = np.isin(y_train, labels)

    # Re-map the labels and X matrices for each of the parents in the tree
    X_list = [np.empty_like(X_train)] * nkeys
    y_list = [np.empty_like(y_train)] * nkeys
    for (i, key) in enumerate(tree_dict.keys()):
        # Get the new X matrix
        X_new = X_train[idx[i], :]
        X_list[i] = X_new

        # Re-map the labels
        y_new = y_train[idx[i]]
        y_new = remap_labels(y_new, tree_dict[key])
        y_list[i] = y_new

    # Train each of the parent nodes in parallel
    start_time = time()
    niter_rep = repeat(niter)
    rng_rep = repeat(rng)
    estimator_rep = repeat(estimator)
    method_rep = repeat("db2")
    with Pool() as p:
        models = p.starmap(train_node, zip(X_list, y_list, niter_rep, rng_rep,
                                           estimator_rep, method_rep))
    train_time = time() - start_time

    # Update the models object to have the key --> clf
    clfs = [models[i]["clf"] for i in range(len(models))]
    model_dict = dict(zip(tree_dict.keys(), clfs))

    # Get the out-of-sample predictions
    y_pred = np.empty(shape=(X_test.shape[0]), dtype=int)
    idx = np.arange(X_test.shape[0])
    label_group = str(np.unique(y_train).tolist())
    y_pred = db2_pred(X_test, label_group, y_pred, tree_dict, model_dict, idx)

    return {"train_time": train_time, "y_pred": y_pred}


def db2_pred(X: np.ndarray, label_group: str, y_pred: np.ndarray,
             tree_dict: dict, model_dict: dict, idx: np.ndarray) -> np.ndarray:
    """
    Gets the predictions for the DB2 benchmark
    """

    # Get the predictions for the given label group and data
    pred = model_dict[str(label_group)].predict(X[idx, :])

    # Since we are restricting ourselves to binary the resulting output
    # will be {0, 1}
    unique_pred = np.unique(pred)
    for val in unique_pred:
        pred_idx = idx[np.where(pred == val)]

        # If we are at the case where the child of the given label_group
        # parent is a singleton, we know we can make the final prediction
        if len(tree_dict[label_group][val]) == 1:
            # Update the final prediction values
            y_pred[pred_idx] = tree_dict[label_group][val][0]
        else:
            # We are in a case where we are not dealing with a singleton
            # and thus need to recurse down the DB2 tree
            new_label_group = str(tree_dict[label_group][val])
            y_pred = db2_pred(X[idx, :], new_label_group, y_pred,
                              tree_dict, model_dict)

    return y_pred
