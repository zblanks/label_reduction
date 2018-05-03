import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool


def cluster_data(X, k):
    """Clusters our data via k-means

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        k (int): Number of clusters to find

    Returns:
        float: Total cluster dispersion for the given value of k
    """
    # Define our KMeans object
    kmeans = KMeans(n_clusters=k, random_state=17)

    # Perform the clustering and return the total cluster
    # dispersion
    kmeans.fit(X=X)
    return kmeans.inertia_


def gen_reference_distn(X):
    """Generates our reference distribution using the simple
    uniform sampling scheme

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data

    Returns:
        ndarray, shape = [n_samples, n_features]: Reference data
    """

    # Define a matrix placeholder where we can store our data
    n, p = X.shape
    Z = np.empty(shape=(n, p))

    # Go through each feature, determine its range, and then sample uniformly
    # over that range
    for j in range(p):
        # Get the max and min of the values of that feature
        lwr = np.min(X[:, j])
        upr = np.max(X[:, j])
        tmp_data = np.random.uniform(low=lwr, high=upr, size=n)

        # Add our data to our reference distribution
        Z[:, j] = tmp_data

    return Z


def bootstrap_data(X):
    """Bootstraps our data for the Monte Carlo simulation of E[log(W)]

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data

    Returns:
        ndarray, shape = [n_samples, n_features]: Bootstrapped version of X
    """

    # Generate bootstrap indexes
    n = X.shape[0]
    idx = np.random.choice(a=np.arange(n), size=n, replace=True)

    # Get our bootstrapped data
    return X[idx, :]


def get_dispersion(X, k_range, n_boot):
    """Gets our dispersion value for the true data and the reference
    distribution for every value of k using n_boot replications

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        k_range (ndarray, shape = [k_range, ]): List of cluster values to
        search over
        n_boot (int): Number of bootstrap iterations to perform

    Returns:
        ndarray, shape = [n_boot, k_range] = Bootstrapped dispersion value
        for the reference distribution

        ndarray, shape = [k_range, 1] = True dispersion values for every k
    """

    # Generate our reference distribution
    Z = gen_reference_distn(X=X)

    # Define a matrix to hold our bootstrap values
    W = np.empty(shape=(n_boot, len(k_range)))

    # Get our bootstrap reference values
    for i in tqdm(range(n_boot)):

        # Generate a bootstrapped data from the reference distribution
        B = bootstrap_data(X=Z)

        # Get the dispersion from the bootstrapped reference distribution
        with Pool() as p:
            tmp_dispersion = p.starmap(cluster_data, zip([B] * len(k_range),
                                                         k_range))

        # Add the results to our W matrix
        W[i, :] = tmp_dispersion

    # Compute the dispersion for our true data
    with Pool() as p:
        true_dispersion = p.starmap(cluster_data, zip([X] * len(k_range),
                                                      k_range))
    true_dispersion = np.array(true_dispersion).reshape(-1, 1)
    return W, true_dispersion


def compute_gap(W, dispersion):
    """Computes the gap statistic for every value of k = 1, ..., K and
    the corresponding standard deviation

    Args:
        W (ndarray shape = [n_boot, k_range]: Bootstrapped dispersion values
        dispersion (ndarray shape = [k_range, 1]: True dispersion values

    Returns:
        ndarray, shape = [k_range,]: Gap stat for every k = 1, ..., K
        ndarray, shape = [k_range,]: sd_k * sqrt(1 + 1/B) for all k
    """

    # First we need to make our dispersion into a (n_boot, k_range) matrix
    # so that we can perform computations efficiently
    D = np.repeat(dispersion.T, repeats=W.shape[0], axis=0)

    # Now we can compute the gap statistic for our system
    W = np.log(W)
    D = np.log(D)
    G = W - D
    G = G.mean(axis=0)

    # Finally we need the standard deviation to the select the best k
    sd = W.std(axis=0)
    s = sd * np.sqrt(1 + (1 / W.shape[0]))
    return G, s


def get_best_k(G, s, k_range):
    """Determines the best K value given the gap statistic and the
    standard deviation

    Args:
        G (ndarray, shape = [k_range, 1]: Gap stat for k = 1, ..., K
        s (ndarray, shape = [k_range, 1]: sd_k * sqrt(1 + 1/B) for all k
        k_range (ndarray, shape = [k_range, ]): List of cluster values to
        search over

    Returns:
        int: Best K value according to the gap statistic
    """

    # We need to find the smallest k such that Gap(k) >= Gap(k + 1) - s_k
    n_clust = G.shape[0]
    for i in range(n_clust):
        if i == n_clust:
            print('Returning largest value of k: {}; consider increasing'
                  'your range of k'.format(i))
            return k_range[i]
        else:
            if G[i] > G[i + 1] - s[i]:
                return k_range[i]


def run_gap_stat(X, k_range, n_boot):
    """Runs the gap stat algorithm and finds the best value of k over
    a pre-specified range

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        k_range (ndarray, shape = [k_range, ]): List of cluster values to
        search over
        n_boot (int): Number of bootstrap iterations to perform

    Returns:
        int: Best value of k according to the gap stat
    """

    # Compute our dispersion values over the specified range of k
    W, dispersion = get_dispersion(X=X, k_range=k_range, n_boot=n_boot)

    # Using our dispersion values, compute the gap statistic
    G, s = compute_gap(W=W, dispersion=dispersion)

    # Finally determine the best value for k given the gap stat and
    # the computed standard deviation
    best_k = get_best_k(G=G, s=s, k_range=k_range)
    return best_k


def save_gap_res(X, k_range, n_boot, path):
    """Saves our gap stat results and allows us to visualize what we're doing

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        k_range (ndarray, shape = [k_range, ]): List of cluster values to
        search over
        n_boot (int): Number of bootstrap iterations to perform
        path (str): Path to save the gap results

    Returns:
        Nothing; just saves the object to disk
    """
    # Compute our dispersion values over the specified range of k
    W, dispersion = get_dispersion(X=X, k_range=k_range, n_boot=n_boot)

    # Using our dispersion values, compute the gap statistic
    G, s = compute_gap(W=W, dispersion=dispersion)

    # Finally determine the best value for k given the gap stat and
    # the computed standard deviation
    best_k = get_best_k(G=G, s=s, k_range=k_range)

    # Save our results to disk so that we can visualize them with R
    df = pd.DataFrame({'gap': G, 'std': s, 'k': k_range})
    df.to_csv(path, index=False)
