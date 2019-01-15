import numpy as np
from scipy.stats import entropy
import re
import pandas as pd
import os
from joblib import Parallel, delayed
from sklearn.metrics import log_loss


def get_id(file: str) -> str:
    """
    Gets the unique IDs from the file string
    """
    file_base = os.path.basename(file)
    return re.sub(r"hc_|f_|.csv", "", file_base)


def compute_entropy(P: np.ndarray, Y: np.ndarray, idx_list: list,
                    run_id: str) -> pd.DataFrame:
    """
    Computes the median entropy and log loss for each label and for the entire
    matrix
    """

    # Compute the entropy of each Pr(y|x) for each sample
    entropy_vals = np.apply_along_axis(entropy, 1, P)

    # Compute the median entropy value for each of the indices that correspond
    # to the unique labels in the data
    n = len(idx_list)
    med_entropy = np.empty(shape=(n+1,), dtype=np.float64)
    for i in range(n):
        med_entropy[i] = np.median(entropy_vals[idx_list[i]])

    # And finally compute the median entropy for the entire matrix
    med_entropy[-1] = np.median(entropy_vals)

    # Similarly compute the log loss for each label and for the entire system
    loss_arr = np.empty(shape=(n+1,), dtype=np.float64)
    for i in range(n):
        try:
            loss_arr[i] = log_loss(Y[idx_list[i]], P[idx_list[i], :])
        except ValueError:
            loss_arr[i] = np.nan

    try:
        loss_arr[-1] = log_loss(Y, P)
    except ValueError:
        loss_arr[-1] = np.nan

    # Compute information about the confidence of the arg-max leaf predictions
    prob_conf = np.empty(shape=(n+1,), dtype=np.float64)
    for i in range(n):
        prob_conf[i] = np.median(P[idx_list[i], i])

    prob_conf[-1] = np.median(P.max(axis=1))

    # Put the results in a DataFrame to work with them more easily
    uniq_labels = np.arange(n)
    uniq_labels = np.append(uniq_labels, ["all"])

    return pd.DataFrame(
        {"id": run_id, "label": np.tile(uniq_labels, 3),
         "metric": np.repeat(["med_entropy", "log_loss", "prob_conf"], n+1),
         "value": np.concatenate((med_entropy, loss_arr, prob_conf))}
    )


def get_leaf_stability(files: list, idx_list: list,
                       Y: np.ndarray) -> pd.DataFrame:
    """
    Gets the entropy results for all of the provided probability prediction
    matrices
    """

    # Get all of the run IDs from the file names
    print("Getting file IDs")
    run_ids = list(map(get_id, files))

    # Get all of the probability matrices and compute the entropy values
    # in parallel to decrease computation time
    print("Getting probability matrices")
    with Parallel(n_jobs=-1, verbose=5) as p:
        P_mats = p(delayed(np.loadtxt)(file) for file in files)

        # Compute the entropy values
        print("Computing entropy and log-loss values")
        entropy_dfs = p(delayed(compute_entropy)(P, Y, idx_list, run_id)
                        for (P, run_id) in zip(P_mats, run_ids))

    # Get the results into one DataFrame
    return pd.concat(entropy_dfs, ignore_index=True)
