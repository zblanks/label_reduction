from keras.applications.xception import Xception
from keras.utils import multi_gpu_model
from glob import glob
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from PIL import Image
from itertools import repeat


class TransformData(object):
    """Transforms the image data to a different feature space using
    the Xception model trained on ImageNet

    Parameters
    ----------
    data_path: str
        Path where the data is located

    n_gpu: int
        Number of GPUs used to make the image predictions

    batch_size: int
        Batch size to make the image transformations

    sample_approx: float
        Percent of the data to use to approximate the mean and standard
        deviation

    reshape_img: bool
        Whether to reshape the images or not

    img_shape: tuple
        Desired image shape

    """

    def __init__(self, data_path, n_gpu=2, batch_size=256, sample_approx=0.10,
                 reshape_img=True, img_shape=(256, 256, 3)):
        self._data_path = data_path
        self._n_gpu = n_gpu
        self._batch_size = batch_size
        self._sample_approx = sample_approx
        self._reshape_img = reshape_img

        # Define the default values for the image data
        self._height, self._width, self._n_channel = img_shape

    def _get_img_files(self):
        """Gets all of the image files and their corresponding labels

        Returns
        -------
        dict
            Dictionary containing the image files and the class labels

        """
        # Get the files and the corresponding labels
        directories = os.listdir(self._data_path)
        img_files = [[]] * len(directories)
        labels = [[]] * len(directories)
        for (i, directory) in enumerate(directories):
            path = os.path.join(self._data_path, directory + '/*')
            img_files[i] = glob(path)
            labels[i] = [i] * len(glob(path))

        # Flatten the list of lists
        img_files = [item for sublist in img_files for item in sublist]
        img_files = np.array(img_files)
        labels = [item for sublist in labels for item in sublist]
        labels = np.array(labels)
        return {'img_files': img_files, 'labels': labels}

    def _define_model(self):
        """Defines the Xception model used to make the predictions

        Returns
        -------
        Xception

        """
        model = Xception(include_top=False,
                         input_shape=(self._height, self._width,
                                      self._n_channel), pooling='max')
        model = multi_gpu_model(model=model, gpus=self._n_gpu)
        return model

    def _get_imgs(self, img_files):
        """Reads in the images from img_files

        Parameters
        ----------
        img_files: np.ndarray

        Returns
        -------
        np.ndarray

        """
        # Read in the subset of images
        with Pool() as p:
            imgs = p.map(Image.open, img_files)

        # Reshape the images if we need to
        if self._reshape_img:
            img_shape = (self._height, self._width, self._n_channel)
            img_shape = repeat(img_shape)
            with Pool() as p:
                imgs = p.starmap(self._create_thumbnail,
                                 zip(imgs, img_shape))

        # Get the image as arrays
        with Pool() as p:
            imgs = p.map(np.array, imgs)
        imgs = np.array(imgs)
        return imgs

    def _image_generator(self, img_files, mean, std, steps):
        """Defines the image generator object we will use to make our
        transformations

        Parameters
        ----------
        img_files: np.ndarray
        mean: float
        std: float
        steps: int
            Number of steps to run the generator to go through all of
            the samples

        Returns
        -------
        np.ndarray
            Yields a batch of images that have been normalized and re-sized

        """
        for i in range(steps):
            # Get the image arrays
            files = img_files[(self._batch_size * i):
                              (self._batch_size * (i+1))]
            imgs = self._get_imgs(files)

            # Standardize the images
            imgs = (imgs - mean) / std
            yield imgs

    @staticmethod
    def _create_thumbnail(img, new_shape):
        """Creates a thumbnail

        Parameters
        ----------
        img: Image.Image

        new_shape: tuple
            Desired new image shape

        Returns
        -------
        Image.Image
            PIL Image object

        """
        img.thumbnail(new_shape)
        return img

    def _compute_mean_std(self, img_files):
        """Approximates the mean and standard deviation for the data which
        allows us to apply this transformation to the images

        Parameters
        ----------
        img_files: np.ndarray
            Array containing all of the image files

        Returns
        -------
        dict
            Dictionary containing the approximated sample mean and standard
            deviation

        """

        # We'll grab a random subset of the data to help us compute the
        # necessary summary statistics
        idx = np.random.choice(a=[True, False], size=len(img_files),
                               replace=True, p=[self._sample_approx,
                                                1 - self._sample_approx])
        files = img_files[idx]
        imgs = self._get_imgs(files)

        # Using the images we will now approximate the sample mean and
        # standard deviation
        sample_mean = np.mean(imgs)
        sample_std = np.std(imgs)
        return {'mean': sample_mean, 'std': sample_std}

    def transform(self):
        """Gets the predictions from the Xception model

        Returns
        -------
        DataFrame
            DataFrame containing the transformed data and each sample's
            corresponding label

        """

        # Get the image files and the corresponding labels
        file_dict = self._get_img_files()

        # Determine the number of steps it will take to get through
        # transforming all of the images
        if len(file_dict['img_files']) % self._batch_size == 0:
            steps = len(file_dict['img_files']) / self._batch_size
        else:
            steps = int(np.floor(len(file_dict['img_files']) /
                                 self._batch_size)) + 1

        # Approximate the mean and standard deviation of the images in
        # the data
        stat_dict = self._compute_mean_std(file_dict['img_files'])

        # Define our image generator
        img_gen = self._image_generator(
            img_files=file_dict['img_files'], mean=stat_dict['mean'],
            std=stat_dict['std'], steps=steps
        )

        # Define the model we will use to transform the data
        model = self._define_model()

        # Get all of the transformations
        data = model.predict_generator(generator=img_gen, verbose=1,
                                       steps=steps)

        # Put the data in the form we expect it to be for the
        # SimilarityMeasure object
        data = pd.DataFrame(data)
        data.loc[:, 'label'] = file_dict['labels']
        return data
