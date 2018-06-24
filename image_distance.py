import numpy as np
import pandas as pd
from imageio import imread
from itertools import combinations, repeat, product
import os
from glob import glob
from multiprocessing import Pool


class ImageDistance(object):
    """Computes the L2 distance for each image in the data set so we can
    compute the class and combination similarity

    Parameters
    ----------
    file_path: str
        File path to the image data set (assumed that it contains a list of
        directories which belong to a class of images; ex: apple)

    n_sample: int
        Number of samples to take to approximate the similarity

    norm: int
        Whether to use the L1 or L2 norm

    Attributes
    ----------
    combo_sim: array
        Pairwise combination array

    class_sim: array
        Lone class similarity array

    """

    def __init__(self, file_path, n_sample=100, norm=2):
        self._file_path = file_path
        self._n_sample = n_sample
        self._norm = norm

        # Define a placeholder for the pairwise and lone class similarity
        # measures
        self.combo_sim = np.array([])
        self.class_sim = np.array([])

        # Define a placeholder for the file DataFrame which maps an image
        # file to a particular label
        self._file_df = pd.DataFrame()

    @staticmethod
    def _flatten_list(x):
        """Flattens a list of lists

        Parameters
        ----------
        x: list

        Returns
        -------
        list

        """
        return [item for sublist in x for item in sublist]

    def _create_file_df(self):
        """

        Returns
        -------
        object: self

        """

        # Get a list of all of the directories in our image data set
        directories = os.listdir(self._file_path)

        # Define placeholders for the files and the labels
        image_files = [[]] * len(directories)
        labels = [[]] * len(directories)
        for (i, directory) in enumerate(directories):
            path = os.path.join(self._file_path, directory + '/*')
            image_files[i] = glob(path)
            labels[i] = [i] * len(image_files[i])

        # Flatten the list of lists so they can be added in a DataFrame
        image_files = self._flatten_list(image_files)
        labels = self._flatten_list(labels)
        self._file_df = pd.DataFrame({'file': image_files, 'label': labels})
        return self

    @staticmethod
    def _rescale_img(img):
        """Re-scales an image to be between [0, 1]

        Parameters
        ----------
        img: array

        Returns
        -------
        array

        """
        return img / 255

    def _compute_sim(self, label):
        """Computes the L2 similarity metric for either a single or
        combination of classes

        Parameters
        ----------
        label
            List of labels either containing a label combination or a lone
            class

        Returns
        -------
        float
            Average lone class similarity for the provided label as indicated
            by the L2 norm

        """
        # Grab the files that correspond to only the provided label
        image_files = self._file_df.loc[(self._file_df.label.isin(label)),
                                        'file']
        image_files = image_files.tolist()
        image_files = np.array(image_files)

        # We have to account for the cases that we're working with a lone
        # class or that we're working with a combination; we do separate
        # things for each of these instances
        if len(label) == 1:
            random_files = np.random.choice(np.arange(len(image_files)),
                                            size=self._n_sample)
            image_files = image_files[random_files]

            # Read in the images and scale them between [0, 1]
            images = map(imread, image_files)
            images = list(map(self._rescale_img, images))

            # Compute the image difference for each (n choose 2) image
            # combination
            combos = combinations(range(self._n_sample), 2)
            images = repeat(images)
            image_diff = map(lambda img, idx: img[idx[0]] - img[idx[1]],
                             images, combos)
        else:
            # Grab the files corresponding to the first class
            first_files = self._file_df.loc[(self._file_df.label == label[0]),
                                            'file']
            random_files = np.random.choice(first_files.index.tolist(),
                                            size=(self._n_sample // 2),
                                            replace=False)
            first_files = first_files.loc[random_files]

            # Grab the files corresponding to the second class
            second_files = self._file_df.loc[(self._file_df.label == label[1]),
                                             'file']
            random_files = np.random.choice(second_files.index.tolist(),
                                            size=(self._n_sample // 2),
                                            replace=False)
            second_files = second_files.loc[random_files]

            # Read and scale for both cases
            first_imgs = map(imread, first_files)
            first_imgs = map(self._rescale_img, first_imgs)
            second_imgs = map(imread, second_files)
            second_imgs = map(self._rescale_img, second_imgs)

            # We need to create every possible pair of first and second
            # images so that we can compute the pairwise value
            img_prod = product(first_imgs, second_imgs)

            # Use our list of image tuples corresponding to the first and
            # second classes, we need to compute their difference
            image_diff = map(lambda x: x[0] - x[1], img_prod)

        # Compute the image norm
        if self._norm == 2:
            diff_norm = list(map(np.linalg.norm, image_diff))
        else:
            diff_norm = list(map(lambda x: np.abs(x).sum(), image_diff))
        return np.array(diff_norm).mean()

    def run(self):
        """Runs the ImageDistance object and computes the L2 similarity
        for all classes and pairwise class combinations

        Returns
        -------
        object: self

        """

        # First get all of the relevant files and put them in the file
        # DataFrame
        self._create_file_df()

        # Generate the list of lists for the lone classes
        unique_labels = np.unique(self._file_df.label)
        labels = [[]] * len(unique_labels)
        for i in range(len(unique_labels)):
            labels[i] = [i]
        with Pool() as p:
            self.class_sim = p.map(self._compute_sim, labels)

        # Generate all pairwise label combos
        combos = combinations(unique_labels, 2)
        combos = list(map(list, combos))
        with Pool() as p:
            self.combo_sim = p.map(self._compute_sim, combos)

        # Convert the measures into an array and return the object
        self.class_sim = np.array(self.class_sim)
        self.combo_sim = np.array(self.combo_sim)
        return self
