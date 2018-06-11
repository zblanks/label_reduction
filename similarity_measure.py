from sklearn.metrics.pairwise import pairwise_kernels
from scipy.special import comb
import pandas as pd
import numpy as np
from itertools import combinations
from multiprocessing import Pool


class SimilarityMeasure(object):
    """Class to compute the similarity metric and compute the relevant
    correction factor

    Parameters
    ----------
    df: DataFrame
        Data (with labels) used to compute the similarity for pairs or
        combinations

    metric: str
        Similarity metric

    combo_approx_num: int
        Number of combinations to use for the correction factor

    compute_correction: bool
        Whether we will compute the correction factor

    Attributes
    ----------
    sim_mat_: array, shape=(n_sample, n_sample)
    class_sim_: array, shape=(n_class, 1)
    combo_sim_: array, shape=(n_combo, 1)
    correction_factor_: dict

    """

    def __init__(self, df, metric, combo_approx_num, compute_correction=False):
        self.df = df
        self.metric = metric
        self.combo_approx_num = combo_approx_num
        self.compute_correction = compute_correction

        # Determine the number of classes in the data
        self.n_class = len(np.unique(self.df.label))
        self.n_combo = int(comb(self.n_class, 2))

        # Get our pairwise combinations
        self.combos = list(combinations(range(self.n_class), 2))

        # Initialize the relevant attributes
        self.sim_mat_ = np.empty(shape=(self.df.shape[0], self.df.shape[0]))
        self.class_sim_ = np.empty(shape=(self.n_class, 1))
        self.combo_sim_ = np.empty(shape=(self.n_combo, 1))

        # Get every possible combination value to instantiate our
        # correction factor dictionary
        self.correction_factor_ = {}

    def get_idx(self):
        """Gets our combination and class indexes

        Returns
        -------
        list: class_idx
        list: combo_idx

        """

        # First get the indexes for each of our classes
        class_idx = [None] * self.n_class
        for i in range(self.n_class):
            class_idx[i] = self.df.index[self.df.label == i].tolist()

        # Now get the combination indexes
        combo_idx = [None] * self.n_combo
        for (i, combo) in enumerate(self.combos):
            combo_idx[i] = class_idx[combo[0]] + class_idx[combo[1]]
        return class_idx, combo_idx

    def compute_sim_matrix(self):
        """Computes the similarity matrix for the provided data

        Returns
        -------
        array, shape=(n_sample, n_sample): Kernel similarity matrix

        """

        # Transform our data into an array
        data = self.df.drop(['label'], axis=1)
        data = data.as_matrix()

        # Compute the similarity metric
        sim_mat = pairwise_kernels(X=data, metric=self.metric, n_jobs=-1)
        return sim_mat

    def compute_similarity(self, idx_list):
        """Computes the similarity for either the single or pairwise
        approximation

        Parameters
        ----------
        idx_list: list of lists containing the indexes for class/combos

        Returns
        -------
        array, shape=(n_class or n_combo, )
            Similarity vector by class or combination

        """

        # Compute the similarity by grabbing only the relevant indexes
        # and then taking their average
        sim = np.empty(shape=(len(idx_list)))
        for (i, idx) in enumerate(idx_list):
            tmp_mat = self.sim_mat_[np.ix_(idx, idx)]
            tmp_idx1, tmp_idx2 = np.tril_indices_from(arr=tmp_mat, k=-1)
            sim[i] = tmp_mat[tmp_idx1, tmp_idx2].mean()
        return sim

    def get_similarity_measures(self):
        """Computes the lone and pairwise approximation similarity measures
        and returns their arrays

        Returns
        -------
        object: self

        """

        # Re-map our labels to be 0, ..., C
        labels = np.unique(self.df.label)
        label_map = dict(zip(labels, range(len(labels))))
        self.df.label = np.vectorize(label_map.get)(self.df.label)

        # Get the class and combination indexes
        class_idx, combo_idx = self.get_idx()

        # Compute the similarity matrix
        self.sim_mat_ = self.compute_sim_matrix()

        # Compute the similarity vectors
        self.class_sim_ = self.compute_similarity(idx_list=class_idx)
        self.combo_sim_ = self.compute_similarity(idx_list=combo_idx)
        return self

    def find_pair_idx(self, pair):
        """Finds the index of the particular (i, j) combo so that we can
        extract it from our pairwise similarity vector

        Parameters
        ----------
        pair: tuple
            (i, j) combination

        Returns
        -------
        int: index where (i, j) is located

        """
        combo_arr = np.array(self.combos)
        return np.where((combo_arr[:, 0] == pair[0]) &
                        (combo_arr[:, 1] == pair[1]))[0].tolist()

    def compute_pairwise_approx(self, group):
        """Computes the pairwise approximation for the overall group similarity

        Parameters
        ----------
        group: list
            group of classes we're interested in (ex: [0, 1, 2])

        Returns
        -------
        float:
            Pairwise approximation of the group similarity

        """
        # Get all of the relevant pairwise indexes for the group
        group_combos = list(combinations(group, 2))
        group_idx = [0] * len(group_combos)
        for (i, pair) in enumerate(group_combos):
            group_idx[i] = self.find_pair_idx(pair)

        # Compute the pairwise approximation
        group_idx = self.flatten_list(group_idx)
        return self.combo_sim_[group_idx].mean()

    def get_label_idx(self, group):
        """Gets the indexes which correspond to a group of labels

        Parameters
        ----------
        group: list of classes of interest

        Returns
        -------
        list:
            List of label indexes

        """
        return self.df.label.index[self.df.label.isin(group)].tolist()

    def compute_true_group_similarity(self, label_idx):
        """

        Parameters
        ----------
        label_idx: list
            List of label indexes for the classes of interest

        Returns
        -------
        float:
            True group similarity value

        """
        mat = self.sim_mat_[np.ix_(label_idx, label_idx)]
        idx1, idx2 = np.tril_indices_from(arr=mat, k=-1)
        return mat[idx1, idx2].mean()

    @staticmethod
    def flatten_list(x):
        """Flattens a list of lists

        Parameters
        ----------
        x: list

        Returns
        -------
        list

        """
        return [item for sublist in x for item in sublist]

    def compute_correction_factor(self):
        """Computes the correction factor

        Returns
        -------
        object: self

        """
        # Define list of lists which will allow us to hold data to
        # eventually compute the correction for all relevant group sizes
        true_sim = [[]] * len(range(3, self.n_class))
        approx_sim = [[]] * len(range(3, self.n_class))
        group_size = [[]] * len(range(3, self.n_class))

        # Now we need to go through every possible group size we could see
        # and compute the correction factor
        count = 0
        for i in range(3, self.n_class):
            combo_iterator = combinations(range(self.n_class), i)
            group_combos = [()] * self.combo_approx_num
            for j in range(self.combo_approx_num):
                group_combos[j] = next(combo_iterator)
            group_size[count] = [count] * self.combo_approx_num

            # Compute the pairwise approximation for the provided group
            # combinations
            with Pool() as p:
                pairwise_approx = p.map(self.compute_pairwise_approx,
                                        group_combos)
            approx_sim[count] = pairwise_approx

            # Get all of the labels for the provided combinations to use to
            # compute the true similarity value
            with Pool() as p:
                group_labels = p.map(self.get_label_idx, group_combos)

            # Using those labels now we can compute the true similarity value
            # (note: we would do this in parallel, but we would like
            # exhaust our memory and thus will have to do it sequentially)
            actual_sim = [0.] * self.combo_approx_num
            for (j, label_idx) in enumerate(group_labels):
                actual_sim[j] = self.compute_true_group_similarity(label_idx)
            true_sim[count] = actual_sim
            count += 1

        # Flatten our data and add it to the DataFrame we will use to compute
        # the correction factor
        true_sim = self.flatten_list(true_sim)
        approx_sim = self.flatten_list(approx_sim)
        group_size = self.flatten_list(group_size)
        correct_df = pd.DataFrame({'true_sim': true_sim,
                                   'approx_sim': approx_sim,
                                   'group_size': group_size})

        # Using our DataFrame we now need to compute the mean absolute
        # deviance of the true versus the approximated similarity value
        # for each group size ranging from {3, ..., C-L}; we note that
        # there is no error for labels which have no combinations or only
        # one combination
        correct_df['sim_diff'] = np.abs(correct_df.true_sim -
                                        correct_df.approx_sim)
        correct_df.drop(['approx_sim', 'true_sim'], axis=1, inplace=True)
        correct_df = correct_df.groupby('group_size').mean()
        correct_df.index = range(3, self.n_class)
        self.correction_factor_ = correct_df.to_dict()['sim_diff']

        # By construction there is no correction if we have a group of size
        # one or two in a label and thus we set these to zero
        self.correction_factor_[1] = 0.
        self.correction_factor_[2] = 0.
        return self

    def run(self):
        """Runs the SimilarityMeasure class and computes the
        class and pairwise similarity and also computes the correction factor

        Returns
        -------
        object: self

        """
        self.get_similarity_measures()
        if self.compute_correction:
            self.compute_correction_factor()
        return self
