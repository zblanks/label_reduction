import numpy as np
from . import _coord_desc
from sklearn.cluster.k_means_ import _k_init
from joblib import Parallel, delayed


def _gen_feasible_soln(V: np.ndarray, seed: int, nclasses: int,
                       nlabels: int, init_algo: str) -> tuple:
    """
    Generates a feasible starting solution for the algorithm
    """

    # The pre-written _k_init seeding algorithm needs the squared norm
    # of the V matrix to work
    if init_algo == "kmeans++":
        V_squared_norms = np.einsum('ij,ij->i', V, V)
        centers = _k_init(V, nlabels, V_squared_norms,
                          np.random.RandomState(seed))
    else:
        rng = np.random.RandomState(seed)
        start_centers = rng.choice(np.arange(nclasses), size=nlabels)
        centers = V[start_centers, :]

    # Using the starting centers, determine the labels ought to be placed
    # with one another by checking which center minimizes the L2 distance
    # from the sample
    z = _coord_desc._get_best_start_centers(V, centers, nclasses)

    # Finally using the initial label assignments, z, we need to re-compute
    # the centroids to determine where the starting centroid values are
    mu = np.empty_like(centers)
    for i in range(nlabels):
        mu[i, :] = V[np.where(z == i)[0], :].mean(axis=0)

    return z, mu


def coord_desc(V: np.ndarray, label_vars: np.ndarray, y: np.ndarray, seed: int,
               nlabels: int, tol=1e-4, max_iter=1000, init_algo="kmeans++"):
    """
    Main function to run the coordinate descent heuristic one time
    """

    # To generate a starting feasible solution, we will use the k-means++
    # heuristic which has good reported performance in practice
    z, mu = _gen_feasible_soln(V, seed, len(np.unique(y)), nlabels, init_algo)

    # Now that we have a starting feasible solution we can perform the
    # coordinate descent search to get the locally optimal solution
    return _coord_desc._run_search(V, z, mu, label_vars, tol, max_iter, seed)


def run_coord_desc(V: np.ndarray, label_vars: np.ndarray, y: np.ndarray,
                   k: int, ninit=10) -> np.ndarray:
    """
    Runs the coordinate descent heuristic ninit times and returns the best
    solution
    """

    # Define placeholders for the results
    z_res = np.empty(shape=(ninit, V.shape[0]), dtype=np.int32)
    obj_val_res = np.empty(shape=(ninit,), dtype=np.float64)

    # Run the algorithm ninit times
    with Parallel(n_jobs=-1) as p:
        res = p(delayed(coord_desc)(V, label_vars, y, i, k)
                for i in range(ninit))

    # Get the best result
    for i in range(ninit):
        z_res[i, :] = res[i][0]
        obj_val_res[i] = res[i][1]

    best_soln = obj_val_res.argmin()
    return z_res[best_soln, :]
