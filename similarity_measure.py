from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
import numpy as np
from itertools import combinations, product, repeat
from joblib import Parallel, delayed


class SimilarityMeasure(object):
    """Class to compute the similarity metric for classes and combinations

    Parameters
    ----------
    path: str
        File path the .h5 file containing the data with the data under the
        key "data" where the column with class labels is the last one

    metric: str
        Similarity metric

    use_dist: bool
        Whether to use a distance or a kernel similarity metric

    sample_approx: float
        Percentage of the data to use for the approximation of the similarity
        values

    chunk_size: int
        Chunk size to use for pool.imap()

    Attributes
    ----------
    class_sim: np.ndarray
    combo_sim: np.ndarray

    """

    def __init__(self, path, metric="euclidean", use_dist=True,
                 sample_approx=0.1, chunk_size=100000):
        self._path = path
        self._metric = metric
        self._use_dist = use_dist
        self._sample_approx = sample_approx
        self._chunk_size = chunk_size

        # Read in the numpy memory map
        self._data = np.load(path, mmap_mode="r")

        # Initialize the relevant attributes
        self.class_sim = np.array([])
        self.combo_sim = np.array([])

        # Get the unique labels in the data
        self._classes = np.unique(self._data[:, -1].astype(int))

    @staticmethod
    def _get_class_idx(label_arr, label):
        """Gets the class indexes

        Parameters
        ----------
        label_arr: np.ndarray

        label: np.ndarray

        Returns
        -------
        combinations
            Combination iterator containing all of the pairwise combos
            for the lone classes

        """
        idx = np.where(label_arr == label)[0]
        return combinations(idx, 2)

    @staticmethod
    def _get_combo_idx(label_arr, combo):
        """Gets the indexes for a particular pairwise combo

        Parameters
        ----------
        label_arr: np.ndarray

        combo: np.ndarray

        Returns
        -------
        product
            Product iterator containing the product of the indexes for
            the first and second label

        """
        # Get the indexes for the first and second label in the combination
        idx1 = np.where(label_arr == combo[0])[0]
        idx2 = np.where(label_arr == combo[1])[0]
        return product(idx1, idx2)

    def _sample_data(self, args_tuple):
        """Samples the data for the given label

        Parameters
        ----------
        args_tuple: tuple
            Tuple whose first argument is the relevant label, the
            second argument is the random seed for the given label, and
            the third argument is the label_array

        Returns
        -------
        np.ndarray
            List of down-sampled indexes

        """
        # Get the indexes with the particular label and grab a random
        # subset of them
        label = args_tuple[0]
        seed = args_tuple[1]
        label_arr = args_tuple[2]
        label_idx = np.where(label_arr == label)[0]
        rng = np.random.RandomState(seed)
        random_entries = rng.choice(a=np.arange(len(label_idx)),
                                    size=np.ceil(self._sample_approx *
                                                 len(label_idx)).astype(int),
                                    replace=False)

        # Down-sample the data
        return label_idx[random_entries]

    @staticmethod
    def _relevant_data(arr, combos):
        """Defines a generator for

        Parameters
        ----------
        arr: np.ndarrray
            Down-sampled data

        combos: combinations
            combinations iterator used to subset the data for the given
            combination

        Returns
        -------
        np.ndarray

        """
        for combo in combos:
            yield arr[np.isin(arr[:, -1], list(combo)), :]

    def _get_sim(self, data):
        """Gets the similarity value for the provided data

        Parameters
        ----------
        data: np.ndarray

        Returns
        -------
        float
            Similarity value

        """
        # Account for the case where we"re working with either lone classes
        # or combinations
        if len(np.unique(data[:, -1])) == 1:
            # Get the label
            label = np.unique(data[:, -1])
            class_idx = self._get_class_idx(data[:, -1], label)

            # Compute the similarity matrix
            sim_mat = self._compute_sim_matrix(data)

            # Compute the similarity value
            return self._compute_similarity(sim_mat, class_idx)
        else:
            # Get the combinations
            combo = np.unique(data[:, -1])
            combo_idx = self._get_combo_idx(data[:, -1], combo)

            # Compute the similarity matrix
            sim_mat = self._compute_sim_matrix(data)

            # Compute the similarity value
            return self._compute_similarity(sim_mat, combo_idx)

    def _compute_sim_matrix(self, arr):
        """Computes the similarity matrix for the provided data

        Parameters
        ----------
        arr: np.ndarray

        Returns
        -------
        np.ndarray
            Similarity matrix

        """

        # Compute the similarity metric
        if self._use_dist:
            sim_mat = pairwise_distances(X=arr[:, :-1], metric=self._metric)
        else:
            sim_mat = pairwise_kernels(X=arr[:, :-1], metric=self._metric)
        return sim_mat

    @staticmethod
    def _compute_similarity(arr, idx):
        """Computes the similarity for either the single or pairwise
        approximation

        Parameters
        ----------
        arr: np.ndarray

        idx: Union[combinations, product]

        Returns
        -------
        float
            Average similarity value

        """
        sample_idx = np.array(list(idx))
        return arr[sample_idx[:, 0], sample_idx[:, 1]].mean()

    def run(self):
        """Runs the SimilarityMeasure class and computes the
        class and pairwise similarity

        Returns
        -------
        object: self

        """

        # Down-sample the data to speed up computation time
        label_arr = self._data[:, -1]
        label_arr_repeat = repeat(label_arr)
        fn_args = zip(self._classes, range(len(self._classes)),
                      label_arr_repeat)
        print("Down-sampling the data")
        with Parallel(n_jobs=-1, backend="threading", verbose=3) as p:
            idx_list = p(delayed(self._sample_data)(arg) for arg in fn_args)

            # Get the down-sampled data
            idx = np.concatenate(idx_list)
            arr = self._data[np.sort(idx), :]

            # Compute the class similarity values
            print("Computing class similarity values")
            combos = combinations(range(len(self._classes)), 1)
            class_data = self._relevant_data(arr, combos)
            class_sim = p(delayed(self._get_sim)(data) for data in class_data)

            # Compute the combination similarity values
            print("Computing combination similarity values")
            combos = combinations(range(len(self._classes)), 2)
            combo_data = self._relevant_data(arr, combos)
            combo_sim = p(delayed(self._get_sim)(data) for data in combo_data)

        self.class_sim = np.array(class_sim)
        self.combo_sim = np.array(combo_sim)
        return self
