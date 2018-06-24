from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
import numpy as np
import pandas as pd
from itertools import combinations, product, repeat
from multiprocessing import Pool


class SimilarityMeasure(object):
    """Class to compute the similarity metric for classes and combinations

    Parameters
    ----------
    df: pd.DataFrame
        Data (with labels) used to compute the similarity for pairs or
        combinations; it is expected that the labels have a column titled
        "label"

    metric: str
        Similarity metric

    use_dist: bool
        Whether to use a distance or a kernel similarity metric

    sample_approx: float
        Percentage of the data to use for the approximation of the similarity
        values

    Attributes
    ----------
    class_sim: np.ndarray
    combo_sim: np.ndarray

    """

    def __init__(self, df, metric='euclidean', use_dist=True,
                 sample_approx=0.10):
        self._df = df
        self._metric = metric
        self._use_dist = use_dist
        self._sample_approx = sample_approx

        # Initialize the relevant attributes
        self.class_sim = np.array([])
        self.combo_sim = np.array([])

        # Check to make sure self._df is a DataFrame
        self._df = pd.DataFrame(self._df)

    def _get_class_idx(self, label):
        """Gets the class indexes

        Parameters
        ----------
        label: int

        Returns
        -------
        np.ndarray
            Array containing the (i, j) index for each intra-class index
            combination used to compute the similarity metric

        """
        indexes = self._df.index[self._df.label == label].tolist()
        idx_combos = list(combinations(indexes, 2))
        return np.array(idx_combos)

    def _get_combo_idx(self, combo):
        """Gets the indexes for a particular pairwise combo

        Parameters
        ----------
        combo: tuple

        Returns
        -------
        np.ndarray

        """
        idx1 = self._df.index[self._df.label == combo[0]].tolist()
        idx2 = self._df.index[self._df.label == combo[1]].tolist()
        idx_prod = list(product(idx1, idx2))
        return np.array(idx_prod)

    def _get_idx(self):
        """Gets the (i, j) indexes for the classes and combos

        Returns
        -------
        dict
            Dictionary containing the (i, j) index arrays for every class
            and combo

        """

        # Get the indexes for the lone classes
        classes = np.unique(self._df.label)
        with Pool() as p:
            class_idx = p.map(self._get_class_idx, classes)

        # Get the combination indexes
        combos = combinations(classes, 2)
        with Pool() as p:
            combo_idx = p.map(self._get_combo_idx, combos)
        return {'class_idx': class_idx, 'combo_idx': combo_idx}

    def _sample_data(self, label):
        """Samples the data for the given label

        Parameters
        ----------
        label: int

        Returns
        -------
        pd.DataFrame
            Down-sampled DataFrame

        """
        df = self._df[self._df.label == label]
        random_entries = np.random.choice([True, False], size=len(df.index),
                                          replace=True,
                                          p=[self._sample_approx,
                                             1 - self._sample_approx])
        df = df.loc[random_entries, :]
        return df

    def _get_data_approx(self):
        """Gets the approximation of the so that we don't have to compute
        the similarity matrix for all n entries

        Returns
        -------
        pd.DataFrame

        """

        # Get the down-sampled DataFrames for each label
        classes = np.unique(self._df.label)
        with Pool() as p:
            df_list = p.map(self._sample_data, classes)

        # Combine the list of DataFrames and reset the index
        data = pd.concat(df_list)
        return pd.DataFrame(data).reset_index(drop=True)

    def _compute_sim_matrix(self):
        """Computes the similarity matrix for the provided data

        Returns
        -------
        np.ndarray
            Similarity matrix

        """

        # Transform our data into an array
        data = self._df.drop(['label'], axis=1)

        # Compute the similarity metric
        if self._use_dist:
            sim_mat = pairwise_distances(X=data, metric=self._metric,
                                         n_jobs=-1)
        else:
            sim_mat = pairwise_kernels(X=data, metric=self._metric,
                                       n_jobs=-1)
        return sim_mat

    @staticmethod
    def _compute_similarity(sim_mat, idx):
        """Computes the similarity for either the single or pairwise
        approximation

        Parameters
        ----------
        sim_mat: np.ndarray
            Similarity matrix

        idx: np.ndarray
            (i, j) index array for the particular class or combination

        Returns
        -------
        float
            Average similarity value

        """
        return sim_mat[idx[:, 0], idx[:, 1]].mean()

    def run(self):
        """Runs the SimilarityMeasure class and computes the
        class and pairwise similarity

        Returns
        -------
        object: self

        """

        # Down-sample the data to speed up computation time
        self._df = self._get_data_approx()

        # First get the necessary indexes for the classes and combinations
        idx_dict = self._get_idx()

        # Compute the similarity matrix
        sim_mat = self._compute_sim_matrix()

        # Compute the lone class similarity values
        sim_mat = repeat(sim_mat)
        with Pool() as p:
            class_sim = p.starmap(self._compute_similarity,
                                  zip(sim_mat, idx_dict['class_idx']))

        # Compute the combination similarity values
        with Pool() as p:
            combo_sim = p.starmap(self._compute_similarity,
                                  zip(sim_mat, idx_dict['combo_idx']))

        self.class_sim = np.array(class_sim)
        self.combo_sim = np.array(combo_sim)
        return self
