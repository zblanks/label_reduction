from sklearn.metrics.pairwise import pairwise_kernels
import pandas as pd
import numpy as np
from itertools import combinations
import os
import sys


def get_sim_mat(X, kernel):
    """Computes our RBF kernel similarity matrix

    Args:
        X (ndarray, shape = [n_samples, n_features]: Data
        kernel (str): Kernel we're using

    Returns:
        (ndarray, shape = [n_samples, n_samples]): Kernel similarity matrix
    """
    # Get our similarity matrix
    sim_mat = pairwise_kernels(X=X, metric=kernel, n_jobs=-1)
    return sim_mat


def get_class_sim(sim_mat, class_idx):
    """Computes the in-class similarity vector

    Args:
        sim_mat (ndarray, shape=[n_samples, n_samples]): Similarity matrix
        class_idx (list): List of lists containing each class' indexes

    Returns:
        (ndarray, shape=[C,]): Similarity vector by class
    """
    class_sim = np.empty(shape=(len(class_idx)))
    for (i, idx) in enumerate(class_idx):
        class_sim[i] = sim_mat[np.ix_(idx, idx)].mean()
    return class_sim


def get_comb_sim(sim_mat, comb_idx):
    """Computes the similarity of all of our class combinations

    Args:
        sim_mat (ndarray, shape=[n_samples, n_samples]): Similarity matrix
        comb_idx (list): List of lists containing the indexes for each class
                         combination

    Returns:
        (ndarray, shape=[choose(C, 2), ]): Similarity vector for each
        class combination
    """

    comb_sim = np.empty(shape=(len(comb_idx)))
    for (i, idx) in enumerate(comb_idx):
        comb_sim[i] = sim_mat[np.ix_(idx, idx)].mean()
    return comb_sim


def get_idx(df):
    """Gets our combination and class indexes

    Args:
        df (DataFrame): Data with labels

    Returns:
        list: List of lists for the class indexes
        list: List of lists for the combination indexes
    """

    # Generate our class combinations
    n_class = len(np.unique(df.label))
    combos = list(combinations(np.arange(n_class), 2))

    # First get the indexes for each of our classes
    class_idx = [None] * n_class
    for i in range(n_class):
        class_idx[i] = df.index[df.label == i].tolist()

    # Now get the combination indexes
    combo_idx = [None] * len(combos)
    for (i, combo) in enumerate(combos):
        combo_idx[i] = class_idx[combo[0]] + class_idx[combo[1]]

    return class_idx, combo_idx


def get_measures(path, save_loc, kernel):
    """Uses all of the helper functions to get our similarity measures
    for the IP

    Args:
        path (str): Path to the data file
        save_loc (str): File save location
        kernel (str): Kernel we're using

    Returns:
        Nothing -- saves the vectors to disk
    """

    # Read in our data
    df = pd.read_csv(path)

    # Re-map our labels to be 0, ..., C
    labels = np.unique(df.label)
    label_map = dict(zip(labels, range(len(labels))))
    df.label = np.vectorize(label_map.get)(df.label)

    # Get the class and combination indexes
    class_idx, comb_idx = get_idx(df=df)

    # Compute the similarity matrix
    df.drop(['label'], axis=1, inplace=True)
    data = df.as_matrix()
    sim_mat = get_sim_mat(X=data, kernel=kernel)

    # Compute the similarity vectors
    class_sim = get_class_sim(sim_mat=sim_mat, class_idx=class_idx)
    comb_sim = get_comb_sim(sim_mat=sim_mat, comb_idx=comb_idx)

    # Save the vectors to disk
    np.savetxt(fname=os.path.join(save_loc, 'class_sim.csv'), X=class_sim,
               delimiter=',')
    np.savetxt(fname=os.path.join(save_loc, 'comb_sim.csv'), X=comb_sim,
               delimiter=',')


if __name__ == '__main__':
    file_path = sys.argv[1]
    save_path = sys.argv[2]
    sim_metric = sys.argv[3]
    get_measures(path=file_path, save_loc=save_path, kernel=sim_metric)
