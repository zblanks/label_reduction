import numpy as np
import pandas as pd
from multiprocessing import Pool
import os


def init_cluster(X, eta):
    """Initializes our clusters for DP-means

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        eta (float): Cluster penalty

    Returns:
        list: Initial cluster assignments
        ndarray, shape = [n_features, ]: Global mean of data
        float: Initial objective value
    """

    # Get the initial assignments and global means
    init_assign = [0] * X.shape[0]
    global_mean = X.mean(axis=0).reshape(1, -1)

    # Compute our initial loss
    loss = compute_obj(X=X, clust_assign=init_assign, eta=eta)

    return init_assign, global_mean, loss


def check_convergence(X, prev_loss, curr_assign, eta, tol):
    """Checks if method has converged

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        prev_loss (float): Loss at stage t-1
        curr_assign (list): List of cluster assignments at stage t
        eta (float): Cluster penalty
        tol (float, default = 0.001): Convergence criteria tolerance

    Returns:
        float: Loss value
        bool: T/F of whether the method has converged
    """

    # Get our current objective value
    curr_loss = compute_obj(X=X, clust_assign=curr_assign, eta=eta)

    # Check our convergence condition
    if np.abs(curr_loss - prev_loss) < tol:
        return curr_loss, True
    else:
        return curr_loss, False


def compute_obj(X, clust_assign, eta):
    """Computes the clustering objective given the computed distance matrix

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        clust_assign (list): List of cluster assignments by index
        eta (float): Cluster penalty

    Returns:
        float: Objective value
    """

    # Get our distance vector
    mu = compute_clust_mean(X=X, clust_assign=clust_assign)
    dist_mat = compute_dist(X=X, clust_mean=mu)
    df = pd.DataFrame(dist_mat)
    df['label'] = clust_assign
    dist_vec = np.empty(shape=(df.shape[0]))

    # Get the distances which correspond to the cluster assignments
    for (i, val) in enumerate(clust_assign):
        dist_vec[i] = dist_mat[i, val]

    # Return our objective
    return dist_vec.sum() + (eta * dist_mat.shape[1])


def compute_clust_mean(X, clust_assign):
    """Computes the cluster means

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        clust_assign (list): List of cluster assignments by index

    Returns:
        (ndarray, shape = [n_clust, n_features]): Cluster means
    """
    df = pd.DataFrame(X)
    df['label'] = clust_assign
    return df.groupby(by='label').mean().as_matrix()


def compute_dist(X, clust_mean):
    """Computes the squared L2 distance for all samples to the cluster
    centers

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        clust_mean (ndarray, shape = [n_clust, n_features]): Cluster means

    Returns:
        (ndarray, shape = [n_samples, n_clust]): Distance of each sample to
        the respective cluster centroid
    """

    # Instantiate a distance matrix to hold in memory
    n_clust = clust_mean.shape[0]
    n_samples = X.shape[0]
    dist_mat = np.empty(shape=(n_samples, n_clust))

    # Loop through all of the clusters and get their distances
    for i in range(n_clust):

        # Get our temporary cluster centroid for the ith cluster
        tmp_centroid = clust_mean[i, :].reshape(1, -1)
        tmp_centroid = np.repeat(tmp_centroid, repeats=n_samples, axis=0)

        # Compute the squared L2 distance
        tmp_dist = np.square(np.linalg.norm(x=(X - tmp_centroid), ord=2,
                                            axis=1))

        # Add our distance to the distance matrix
        dist_mat[:, i] = tmp_dist
    return dist_mat


def assign_labels(X, eta, clust_mean):
    """Assigns our cluster labels

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        eta (float): Cluster penalty
        clust_mean (ndarray, shape = [n_clust, n_features]): Cluster means

    Returns:
        list: List of cluster assigns after iteration t
        (ndarray, shape = [n_clust, n_features]): Cluster means
    """

    # Define a list to hold our label assignments
    labels = [None] * X.shape[0]

    # Go through all of our samples and either assign them to existing
    # clusters or create new ones
    for (i, sample) in enumerate(X):

        # Compute the distance to the sample
        sample = sample.reshape(1, -1)
        dist = np.square(np.linalg.norm(x=(sample - clust_mean), ord=2,
                                        axis=1))

        # Check if we need to make a new cluster
        if np.min(dist) > eta:

            # Create a new cluster
            clust_mean = np.concatenate((clust_mean, sample), axis=0)
            labels[i] = clust_mean.shape[0] - 1
        else:
            labels[i] = np.argmin(dist)

    return labels, clust_mean


def adjust_labels(clust_assign):
    """Adjusts our label numbers to correspond to the true number of
    clusters

    Args:
        clust_assign (list): List of cluster labels

    Returns:
        list: List of adjusted label assignments by index
    """

    # Create a map to adjust our labels
    uniq_labels = np.unique(clust_assign)
    n_labels = len(uniq_labels)
    label_map = dict(zip(uniq_labels, np.arange(n_labels)))

    # Adjust our labels
    return np.vectorize(label_map.get)(clust_assign)


def dp_means(X, eta, tol, max_iter):
    """Executes our DP-means algorithm

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        eta (float): Cluster penalty
        tol (float): Convergence criteria tolerance
        max_iter (int): Max number of iterations

    Returns:
        list: Final cluster assignments
        float: Objective value
        int: Number of iterations needed to reach convergence

    Raises:
        If max_iter reached, then failure to converge warning will be raised
    """

    # Initialize our clusters
    clust_labels, clust_mean, loss = init_cluster(X=X, eta=eta)

    # Hold how many iterations we've done thus far
    n_iter = 0

    # Iterate until convergence or we fail to do so
    for i in range(max_iter):

        # Update the number of iterations
        n_iter += 1

        # Get the label assignments and the cluster means at stage t
        new_labels, clust_mean = assign_labels(X=X, eta=eta,
                                               clust_mean=clust_mean)

        # Adjust the label values to reflect the true number of clusters
        new_labels = adjust_labels(clust_assign=new_labels)

        # Adjust our cluster means according to our new labels
        clust_mean = compute_clust_mean(X=X, clust_assign=new_labels)

        # Check if we have converged
        prev_loss = loss
        loss, have_converged = check_convergence(
            X=X, prev_loss=prev_loss, curr_assign=new_labels, eta=eta,
            tol=tol
        )

        # If we have, break out
        if have_converged is True:
            break

    # Get the number of unique clusters
    n_clust = len(np.unique(new_labels))

    # Check if we failed to converge
    if n_iter == max_iter:
        print('Failed to converge in {} steps. Consider increasing the '
              'number of iterations'.format(max_iter))
        return n_clust, loss, n_iter
    else:
        return n_clust, loss, n_iter


def parallel_dp_means(X, eta_vals, tol=1e-4, max_iter=1000):
    """Runs our DP-means algorithm in parallel by searching over a range
    of eta values to get the best assignment

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data matrix
        eta_vals (ndarray, shape = [n_vals, ]): List of cluster penalty values
        tol (float): Convergence criteria tolerance
        max_iter (int): Max number of iterations

    Returns:
        list: Labels with lowest objective value
    """

    # Search over our values of eta to get the best assignments
    n = len(eta_vals)
    with Pool() as p:
        dp_res = p.starmap(dp_means, zip([X] * n, eta_vals, [tol] * n,
                                         [max_iter] * n))

    # # Find the model which corresponds to the best results and then
    # # return those labels
    # dp_res = np.array(dp_res)
    # best_model = np.argmin(dp_res[:, 1])
    # return dp_res[best_model, 0]
    dp_res = np.array(dp_res)
    eta_vals = eta_vals.reshape(-1, 1)
    dp_res = np.concatenate((dp_res, eta_vals), axis=1)
    return dp_res
