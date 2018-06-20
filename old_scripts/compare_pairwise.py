import numpy as np
from itertools import combinations
import pandas as pd
from sklearn.metrics import pairwise_kernels
import argparse
import os
from multiprocessing import Pool
import time


def find_pair_idx(combos, pair):
    """Finds the index of the particular (i, j) combo so that we can
    extract it from our pairwise similarity measures

    Args:
        combos (list): List of pairwise combinations
        pair (tuple): (i, j) combination

    Returns:
        int: Index where condition is true
    """
    return [i for i in range(len(combos)) if combos[i] == pair][0]


def flatten_list(x):
    """Flattens a list of lists

    Args:
        x (list): List of lists to flatten

    Returns:
        list: Flattened list
    """
    return [item for sublist in x for item in sublist]


def get_label_idx(label_data, group):
    """Gets the label indexes for the members of the group

    Args:
        label_data (Series): Series of the label information
        group (list): List of the classes of interest

    Returns:
        list: List of label indexes
    """
    return label_data.index[label_data.isin(group)].tolist()


def compute_pairwise_sim(sim_vect, group, combos):
    """Computes the group similarity from pairwise values

    Args:
        sim_vect (ndarray, shape=[len(combos), ]): List of pairwise similarity
                                                   measures

        group (list): List of the group of classes
        combos (list): List of all (k choose 2) combinations

    Returns:
        float: Pairwise similarity approximation of the overall group
        similarity
    """
    # Get all of the combinations of size from our group
    group_combos = list(combinations(group, 2))
    group_idx = [0] * len(group_combos)
    for (i, pair) in enumerate(group_combos):
        group_idx[i] = find_pair_idx(combos=combos, pair=pair)

    # Compute the average pairwise similarity as the approximation for
    # the overall group similarity
    return sim_vect[group_idx].mean()


def compute_true_group_sim(sim_mat, label_idx):
    """Computes the true value for the group similarity

    Args:
        sim_mat (ndarray, [n_sample, n_sample]): Similarity matrix
        label_idx (list): List of label indexes

    Returns:
        float: True group similarity value
    """
    # Get the subset of the similarity matrix
    mat = sim_mat[np.ix_(label_idx, label_idx)]
    idx1, idx2 = np.tril_indices_from(arr=mat, k=-1)
    return mat[idx1, idx2].mean()


def get_all_sim(label_data, sim_vect, sim_mat, max_group_size, n_combos):
    """Gets both the pairwise approximation and true similarity value
    for all

    Args:
        label_data (Series): Series of the label information
        sim_vect (ndarray, shape=[len(combos), ]): List of pairwise similarity
                                                   measures
        sim_mat (ndarray, [n_sample, n_sample]): Similarity matrix
        max_group_size (int): Max group size to consider
        n_combos (int): Number of combinations to work with

    Returns:
        DataFrame: DataFrame of pairwise and true similarity values up
        to the max group size
    """
    # Compute the number of classes in our data
    n_class = len(np.unique(label_data))

    # Define a list of lists to hold our true, approximated similarity
    # values, and group size for the DataFrame
    true_sim = [[]] * len(range(3, max_group_size + 1))
    approx_sim = [[]] * len(range(3, max_group_size + 1))
    group_size = [[]] * len(range(3, max_group_size + 1))

    # Generate the list of all possible (n_class choose 2) combinations
    # to use with our help function
    combos = list(combinations(range(n_class), 2))

    # Loop through every max group size and compute the pairwise and
    # true group similarity
    j = 0
    for i in range(3, max_group_size + 1):
        # Time our loop performance
        start_time = time.time()

        # Generate all possible groupings and add the information to our
        # group size tracker
        group_iterator = combinations(range(n_class), i)
        group_combos = [()] * n_combos
        for k in range(n_combos):
            group_combos[k] = next(group_iterator)
        tmp_group_size = [i] * n_combos
        group_size[j] = tmp_group_size

        # Compute the pairwise approximation
        with Pool() as p:
            tmp_approx_sim = p.starmap(
                compute_pairwise_sim,
                zip([sim_vect] * n_combos, group_combos,
                    [combos] * n_combos)
            )

        # Add this to our list of lists
        approx_sim[j] = tmp_approx_sim

        # Get all of the labels for the given groups
        with Pool() as p:
            group_labels = p.starmap(
                get_label_idx, zip([label_data] * n_combos,
                                   group_combos)
            )

        # Compute the true group similarity value
        tmp_true_sim = [0.] * len(group_labels)
        for (k, label_idx) in enumerate(group_labels):
            tmp_true_sim[k] = compute_true_group_sim(sim_mat=sim_mat,
                                                     label_idx=label_idx)
        true_sim[j] = tmp_true_sim
        finish_time = time.time() - start_time
        print('Completed iteration {} in {} seconds'.format(j, finish_time))
        j += 1

    # Flatten our similarity and group size lists and it to our final
    # DataFrame
    true_sim = flatten_list(true_sim)
    approx_sim = flatten_list(approx_sim)
    group_size = flatten_list(group_size)
    df = pd.DataFrame({'true_sim': true_sim, 'approx_sim': approx_sim,
                       'group_size': group_size})
    return df


if __name__ == '__main__':
    # Get our scripts arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', help='Working directory for script', type=str)
    parser.add_argument('kernel', help='Kernel for similarity matrix',
                        type=str)
    parser.add_argument('max_group_size', help='Max group size for script',
                        type=int)
    parser.add_argument('n_combos', help='Max number of combos to consider',
                        type=int)
    args = vars(parser.parse_args())

    # Get our data
    data = pd.read_csv(os.path.join(args['wd'], 'fruits_matrix',
                                    'train_encoded.csv'))

    # Grab the label series
    labels = data.label

    # Get the combination similarity data
    sim_vector = np.loadtxt(os.path.join(args['wd'], 'fruits_sim',
                                         'comb_sim.csv'),
                            delimiter=',')

    # Compute the similarity matrix
    similarity_matrix = pairwise_kernels(data.drop(['label'], axis=1),
                                         metric=args['kernel'], n_jobs=-1)

    # Get our results data
    sim_data = get_all_sim(label_data=labels, sim_vect=sim_vector,
                           sim_mat=similarity_matrix,
                           max_group_size=args['max_group_size'],
                           n_combos=args['n_combos'])

    # Save the results to disk
    sim_data.to_csv(os.path.join(args['wd'], 'fruits_res', 'sim_score',
                                 'sim_comparison.csv'), index=False)
