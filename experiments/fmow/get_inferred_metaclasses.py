import pandas as pd
import os
import numpy as np
from sklearn.cluster import SpectralClustering
from joblib import Parallel, delayed
import pickle


def build_cluster_distn(df: pd.DataFrame, label: int, k: int,
                        nlabels: int, epsilon=1e-5) -> pd.DataFrame:
    """
    Builds the cluster distribution for a given label
    """

    # Get the sub-DataFrame that corresponds to the location where
    # we have the given label
    sub_df = df.loc[(df["label"] == label) & (df["k"] == k), :]

    # Get the run_num and group values that correspond to the locations where
    # we have the given label
    vals = sub_df.loc[:, ["run_num", "group"]].apply(lambda x: tuple(x),
                                                     axis=1)
    idx = vals.apply(lambda x: df.index[(df["run_num"] == x[0]) &
                                        (df["group"] == x[1]) &
                                        (df["k"] == k)].values)
    idx = np.concatenate(idx.values)

    # Get the rows where the conditions are met and count the number of
    # times each label coincided
    distn = np.bincount(df.loc[idx, "label"], minlength=nlabels)
    distn = distn / len(vals)

    # To avoid issues of having a disconnected graph, where there is 0 mass,
    # we will add epsilon mass where epsilon is a very small number
    distn[np.where(distn == 0)] = epsilon

    # Get the final result into a DataFrame format so that we can more easily
    # work with it downstream
    col_names = ["k", "label"] + ["val" + str(i) for i in range(nlabels)]
    data = np.concatenate(([k, label], distn)).reshape(1, -1)
    return pd.DataFrame(data=data, columns=col_names)


def build_distns(df: pd.DataFrame, use_meta: int) -> pd.DataFrame:
    """
    Builds all of the empirical cluster distributions for a given experiment
    value
    """
    # Subset the data so we are only working with the appropriate experimental
    # values
    new_df = df.loc[df["use_meta"] == use_meta, :]

    # Get all of the unique label values so we can iterate over them and
    # go through all of the labels and get their distributions
    unique_labels = new_df.label.unique()
    k_vals = new_df.k.unique()
    nlabels = len(unique_labels)
    lab_vals = np.tile(unique_labels, len(k_vals))
    k_vals = np.repeat(k_vals, len(unique_labels))

    with Parallel(n_jobs=-1, verbose=5) as p:
        dfs = p(delayed(build_cluster_distn)(new_df, label, k, nlabels)
                for (label, k) in zip(lab_vals, k_vals))

    return pd.concat(dfs, ignore_index=True)


def build_affinity_mat(df: pd.DataFrame) -> list:
    """
    Builds the affinity matrix for all values of k
    """
    k_vals = df.k.unique()
    distn_arrs = [df.loc[df["k"] == k, :].filter(regex="val.").values
                  for k in k_vals]

    return distn_arrs


def get_clustering(affinity_mat: np.ndarray, k: int,
                   use_meta: int) -> pd.DataFrame:
    """
    Clusters the affinity matrix for a given value of k
    """
    sc = SpectralClustering(n_clusters=k, random_state=17,
                        affinity="precomputed")
    n = affinity_mat.shape[0]
    clusters = sc.fit_predict(affinity_mat)

    # Get the resulting DataFrame in the expected format
    return pd.DataFrame({"k": np.repeat(k, n), "label": np.arange(n),
                         "use_meta": np.repeat(use_meta, n),
                         "group": clusters})


def cluster_affinity_mat(affinity_mats: list, use_meta: int) -> pd.DataFrame:
    """
    Cluster the affinity matrices for all values of k
    """
    # Get the feasible values of k
    k_vals = np.arange(2, affinity_mats[0].shape[0])

    # Go through all feasible values of k and compute the clustering
    with Parallel(n_jobs=-1, verbose=5) as p:
        cluster_dfs = p(delayed(get_clustering)(affinity_mat, k, use_meta)
                        for (affinity_mat, k) in zip(affinity_mats, k_vals))

    return pd.concat(cluster_dfs, ignore_index=True)


def main():
    # Define the main working directory for the script
    os.chdir("/pool001/zblanks/label_reduction_data/fmow")

    # Get the cluster grouping data
    group = pd.read_csv("group_res.csv")
    settings = pd.read_csv("experiment_settings.csv")
    df = pd.merge(group, settings, how="inner", on="id")

    # Compute the cluster distributions for all use_meta cases
    use_meta_vals = df.use_meta.unique()
    distn_dfs = [build_distns(df, val) for val in use_meta_vals]

    # Get all of the affinity matrices
    affinity_mats = [build_affinity_mat(distn_df) for distn_df in distn_dfs]

    # Save the A matrices to disk so we can visualize the spectral clustering
    # later
    with open("a_mats.pickle", "wb") as p:
        pickle.dump(affinity_mats, p)

    # Get the clustering of the affinity matrices for all values of k and
    # use_meta
    cluster_dfs = [
        cluster_affinity_mat(affinity_mat, use_meta)
        for (affinity_mat, use_meta) in zip(affinity_mats, use_meta_vals)
    ]

    # Get the final clustering results
    res_df = pd.concat(cluster_dfs, ignore_index=True)
    res_df.to_csv("label_clustering.csv", index=False)


if __name__ == '__main__':
    main()
