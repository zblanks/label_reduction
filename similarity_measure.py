from sklearn.metrics.pairwise import pairwise_kernels
from scipy.special import comb
import numpy as np
from itertools import combinations


class SimilarityMeasure(object):
    """Class to compute the similarity metric for classes and combinations

    Parameters
    ----------
    df: DataFrame
        Data (with labels) used to compute the similarity for pairs or
        combinations; it is expected that the labels have a column titled
        "label"

    metric: str
        Similarity metric

    Attributes
    ----------
    sim_mat: array, shape=(n_sample, n_sample)
    class_sim: array, shape=(n_class, 1)
    combo_sim: array, shape=(n_combo, 1)

    """

    def __init__(self, df, metric):
        self.df = df
        self.metric = metric

        # Determine the number of classes in the data
        self.n_class = len(np.unique(self.df.label))
        self.n_combo = int(comb(self.n_class, 2))

        # Get our pairwise combinations
        self.combos = list(combinations(range(self.n_class), 2))

        # Initialize the relevant attributes
        self.sim_mat = np.empty(shape=(self.df.shape[0], self.df.shape[0]))
        self.class_sim = np.empty(shape=(self.n_class, 1))
        self.combo_sim = np.empty(shape=(self.n_combo, 1))

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
        object: self

        """

        # Transform our data into an array
        data = self.df.drop(['label'], axis=1)
        data = data.as_matrix()

        # Compute the similarity metric
        self.sim_mat = pairwise_kernels(X=data, metric=self.metric,
                                        n_jobs=-1)
        return self

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
            tmp_mat = self.sim_mat[np.ix_(idx, idx)]
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
        self.compute_sim_matrix()

        # Compute the similarity vectors
        self.class_sim = self.compute_similarity(idx_list=class_idx)
        self.combo_sim = self.compute_similarity(idx_list=combo_idx)
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

    def run(self):
        """Runs the SimilarityMeasure class and computes the
        class and pairwise similarity and also computes the correction factor

        Returns
        -------
        object: self

        """
        self.get_similarity_measures()
        return self
