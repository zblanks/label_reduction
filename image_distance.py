import numpy as np
import pandas as pd
from imageio import imread
from itertools import combinations, repeat
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

    Attributes
    ----------
    combo_sim: array
        Pairwise combination array

    class_sim: array
        Lone class similarity array

    """

    def __init__(self, file_path):
        self._file_path = file_path

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
    def _compute_img_diff(images, idx_combo):
        """Computes the pixel-by-pixel difference between two images

        Parameters
        ----------
        images
            List of images

        idx_combo: tuple
            Index combination

        Returns
        -------
        array

        """
        img1, img2 = idx_combo
        return images[img1] - images[img2]

    def _compute_sim(self, label):
        """Computes the L2 similarity metric for either a single or
        combination of classes

        Parameters
        ----------
        label: list
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

        # Read in all of the provided images
        images = map(imread, image_files)

        # Scale all of the images in the list to be between [0, 1]
        images = list(map(lambda img: img / 255, images))

        # Compute the pixel-by-pixel difference between each image
        combos = combinations(range(len(images)), 2)
        images = repeat(images)
        image_diff = map(self._compute_img_diff, images, combos)

        # Compute the L2 norm for the image difference
        diff_norm = list(map(np.linalg.norm, image_diff))
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
