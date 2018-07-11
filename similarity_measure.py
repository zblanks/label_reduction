from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
import numpy as np
from itertools import combinations, product
from joblib import Parallel, delayed
from numba import njit


def get_similarity_values(data: np.ndarray, use_dist=True, metric="euclidean",
                          sample_approx=0.1) -> dict:
    """Computes the relevant similarity values for all classes and
    combinations
    """

    # Down-sample the data to speed up computation time
    label_arr = data[:, -1]
    classes = np.unique(label_arr)
    print("Down-sampling the data")
    with Parallel(n_jobs=-1, backend="threading", verbose=3) as p:
        idx_list = p(delayed(_sample_data)(i, label_arr, sample_approx)
                     for i in range(len(classes)))

        # Get the down-sampled data
        idx = np.concatenate(idx_list)
        arr = data[np.sort(idx), :]

        # Compute the class similarity values
        print("Computing class similarity values")
        combos = combinations(range(len(classes)), 1)
        class_data = _relevant_data(arr, combos)
        class_sim = p(delayed(_get_sim)(data, use_dist, metric)
                      for data in class_data)

        # Compute the combination similarity values
        print("Computing combination similarity values")
        combos = combinations(range(len(classes)), 2)
        combo_data = _relevant_data(arr, combos)
        combo_sim = p(delayed(_get_sim)(data, use_dist, metric)
                      for data in combo_data)
    return {"class_sim": class_sim, "combo_sim": combo_sim}


@njit
def _compute_similarity(arr: np.ndarray, idx: list) -> float:
    """Computes the similarity for either the single or pairwise
    approximation
    """

    sim_vals = np.empty(shape=len(idx), dtype=np.float64)
    for (i, val) in enumerate(idx):
        sim_vals[i] = arr[val[0], val[1]]
    return sim_vals.mean()


def _relevant_data(arr: np.ndarray, combos: combinations) -> np.ndarray:
    """Defines a generator for only the relevant combination to use
    memory more efficiently
    """
    for combo in combos:
        yield arr[np.isin(arr[:, -1], list(combo)), :]


def _get_combo_idx(label_arr: np.ndarray, combo: np.ndarray) -> product:
    """Gets the indexes for a particular pairwise combo
    """

    # Get the indexes for the first and second label in the combination
    idx1 = np.where(label_arr == combo[0])[0]
    idx2 = np.where(label_arr == combo[1])[0]
    return product(idx1, idx2)


def _get_sim(data: np.ndarray, use_dist: bool, metric: str) -> float:
    """Gets the similarity value for the provided data
    """

    # Account for the case where we"re working with either lone classes
    # or combinations
    if len(np.unique(data[:, -1])) == 1:
        # Get the label
        class_idx = combinations(np.arange(data.shape[0]), 2)

        # Compute the similarity matrix
        sim_mat = _compute_sim_matrix(data, use_dist, metric)

        # Compute the similarity value
        return _compute_similarity(sim_mat, list(class_idx))
    else:
        # Get the combinations
        combo = np.unique(data[:, -1])
        combo_idx = _get_combo_idx(data[:, -1], combo)

        # Compute the similarity matrix
        sim_mat = _compute_sim_matrix(data, use_dist, metric)

        # Compute the similarity value
        return _compute_similarity(sim_mat, list(combo_idx))


def _compute_sim_matrix(data: np.ndarray, use_dist: bool,
                        metric: str) -> np.ndarray:
    """Computes the similarity matrix for the provided data
    """

    # Compute the similarity metric
    if use_dist:
        sim_mat = pairwise_distances(X=data[:, :-1], metric=metric)
    else:
        sim_mat = pairwise_kernels(X=data[:, :-1], metric=metric)
    return sim_mat


def _sample_data(label: int, label_arr: np.ndarray,
                 sample_approx=0.1) -> np.ndarray:
    """Samples the data for the given label
    """

    # Get the indexes with the particular label and grab a random
    # subset of them
    label_idx = np.where(label_arr == label)[0]
    rng = np.random.RandomState(label)
    random_entries = rng.choice(a=np.arange(len(label_idx)),
                                size=np.ceil(sample_approx
                                             * len(label_idx)).astype(int),
                                replace=False)

    # Down-sample the data
    return label_idx[random_entries]
