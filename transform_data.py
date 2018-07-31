from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import multi_gpu_model
from glob import glob
import os
import numpy as np
from PIL import Image
import h5py
from joblib import Parallel, delayed
import re


class TransformData(object):
    """Transforms the image data to a different feature space using
    the Xception model trained on ImageNet

    Parameters
    ----------
    data_path: str
        Path where the data is located

    save_path: str
        Path where to save the .h5 file containing the transformed data

    model_name: str
        Which neural network model to use

    n_gpu: int
        Number of GPUs used to make the image predictions

    batch_size: int
        Batch size to make the image transformations

    sample_approx: float
        Percent of the data to use to approximate the mean and standard
        deviation

    img_shape: tuple
        Desired image shape

    """

    def __init__(self, data_path, save_path, model_name, n_gpu=2,
                 batch_size=256, sample_approx=0.10, img_shape=(256, 256, 3)):
        self._data_path = data_path
        self._save_path = save_path
        self._model_name = model_name
        self._n_gpu = n_gpu
        self._batch_size = batch_size
        self._sample_approx = sample_approx

        # Define the default values for the image data
        self._height, self._width, self._n_channel = img_shape

    @staticmethod
    def _get_label(file: str) -> int:
        """Determines the class label from the file name
        """
        m = re.search("/[0-9]{1,4}/", file)
        m = re.search("[0-9]{1,4}", m.group(0))
        return int(m.group(0))

    def _get_img_files(self) -> dict:
        """Gets all of the image files and their corresponding labels
        """

        # First get all of the files
        path = os.path.join(self._data_path, "**/*.jpg")
        print("Getting all image files")
        img_files = glob(path, recursive=True)
        img_files = np.array(img_files)

        # Using the files, get the class labels
        print("Grabbing image labels")
        with Parallel(n_jobs=1) as p:
            labels = p(delayed(self._get_label)(file) for file in img_files)
        labels = np.array(labels).reshape(-1, 1)
        return {"img_files": img_files, "labels": labels}

    def _define_model(self):
        """Defines the Xception model used to make the predictions
        """
        if self._model_name == "xception":
            model = Xception(include_top=False,
                             input_shape=(self._height, self._width,
                                          self._n_channel), pooling="max")
        elif self._model_name == "densenet":
            model = DenseNet201(include_top=False,
                                input_shape=(self._height, self._width,
                                             self._n_channel), pooling="max")
        elif self._model_name == "inception":
            model = InceptionV3(include_top=False,
                                input_shape=(self._height, self._width,
                                             self._n_channel), pooling="max")
        else:
            model = InceptionResNetV2(include_top=False,
                                      input_shape=(self._height, self._width,
                                                   self._n_channel),
                                      pooling="max")
        model = multi_gpu_model(model=model, gpus=self._n_gpu)
        return model

    @staticmethod
    def _convert_img(img: Image.Image) -> Image.Image:
        """Converts an image from gray-scale to RGB if necessary
        """
        return img.convert("RGB")

    def _get_imgs(self, img_files: np.ndarray) -> np.ndarray:
        """Reads in the images from img_files
        """
        # Read in the subset of images
        with Parallel(n_jobs=1) as p:
            imgs = p(delayed(Image.open)(file) for file in img_files)
            imgs = p(delayed(self._convert_img)(img) for img in imgs)

            # Reshape the images
            img_shape = (self._height, self._width)
            imgs = p(delayed(self._create_thumbnail)(img, img_shape)
                     for img in imgs)

            # Convert the image to numpy arrays
            imgs = p(delayed(np.array)(img) for img in imgs)
        return np.array(imgs)

    def _image_generator(self, img_files: list, mean: float,
                         steps: int) -> np.ndarray:
        """Defines the image generator object we will use to make our
        transformations
        """
        for i in range(steps):
            # Get the image arrays
            files = img_files[(self._batch_size * i):
                              (self._batch_size * (i+1))]
            imgs = self._get_imgs(files)

            # Standardize the images
            yield (imgs - mean)

    @staticmethod
    def _create_thumbnail(img: Image.Image, new_shape: tuple) -> Image.Image:
        """Creates a thumbnail
        """
        return img.resize(new_shape)

    def _compute_running_mean(self, img_files: np.ndarray) -> float:
        """Computes the running mean to
        """
        rng = np.random.RandomState(17)
        idx = rng.choice(np.arange(len(img_files)),
                         size=np.ceil(self._sample_approx
                                      * len(img_files)).astype(int),
                         replace=False)
        files = img_files[idx]

        mu = 0.
        for i, file in enumerate(files):
            if i == 0:
                mu = self._get_imgs(np.array([file])).mean()
            else:
                x = self._get_imgs(np.array([file])).mean()
                mu_prev = mu
                mu = mu_prev + ((x - mu_prev) / (i + 1))
        return mu

    def transform(self) -> np.ndarray:
        """Gets the transformations from the Xception model
        """

        # Get the image files and the corresponding labels
        file_dict = self._get_img_files()

        # Determine the number of steps it will take to get through
        # transforming all of the images
        if len(file_dict["img_files"]) % self._batch_size == 0:
            steps = len(file_dict["img_files"]) / self._batch_size
        else:
            steps = int(np.floor(len(file_dict["img_files"]) /
                                 self._batch_size)) + 1

        # Approximate the mean and standard deviation of the images in
        # the data
        print("Computing image mean")
        mu = self._compute_running_mean(file_dict["img_files"])

        # Define our image generator
        img_gen = self._image_generator(
            img_files=file_dict["img_files"], mean=mu, steps=steps
        )

        # Define the model we will use to transform the data
        model = self._define_model()

        # Get all of the transformations
        print("Transforming image data")
        data = model.predict_generator(generator=img_gen, verbose=1,
                                       steps=steps)

        # Put the data in the form we expect it to be for the
        # SimilarityMeasure object
        data = np.concatenate([data, file_dict["labels"]], axis=1)
        f = h5py.File(self._save_path, "w")
        f.create_dataset("data", data=data)
        f.close()
        return data
