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


class TransformData(object):
    """Transforms the image data to a different feature space using
    the a provided model trained on ImageNet

    Parameters
    ----------
    data_path: str
        Path where the data is located

    save_path: str
        Path where to save the .h5 file containing the transformed data

    model_name: str
        Which neural network model to use

    ngpu: int
        Number of GPUs used to make the image predictions

    batch_size: int
        Batch size to make the image transformations

    img_shape: tuple
        Desired image shape

    """

    def __init__(self, data_path, save_path, model_name, ngpu=2,
                 batch_size=256, img_shape=(256, 256, 3)):
        self._data_path = data_path
        self._save_path = save_path
        self._model_name = model_name
        self._ngpu = ngpu
        self._batch_size = batch_size

        # Define the default values for the image data
        self._height, self._width, self._n_channel = img_shape

    @staticmethod
    def _get_labels(files: np.ndarray) -> np.ndarray:
        """
        Infers the labels from the provided image files
        """

        # We expect a file to have a format:
        # /path/to/file/data/label/filename.jpg and so we need to go two
        # levels up to get a list of all targets in the data
        targets = os.path.dirname(os.path.dirname(files[0]))

        # Using targets we will generate a dictionary like {airport => 0, ...}
        # and this will help us map the names found in the files to their
        # numeric values
        label_dict = dict(zip(targets, range(len(targets))))

        # To determine the label of a target, we expect our directory
        # that has the form
        # airport
        #   ....
        # airport_hangar
        #   ...
        # and so forth
        # therefore we need to determine which folder the file belongs in
        # and using this we can determine its label
        dirs = [os.path.dirname(file) for file in files]
        labels = [os.path.basename(val) for val in dirs]

        # Finally we need to map the labels we just inferred to their numeric
        # values using the dictionary we just created
        return np.array([label_dict[label] for label in labels])

    def _get_img_files(self, **kwargs) -> dict:
        """Gets all of the image files and their corresponding labels
        """

        # First get all of the files
        path = os.path.join(self._data_path, "**/*.jpg")
        print("Getting all image files")
        img_files = glob(path, recursive=True)
        img_files = np.array(img_files)

        # Using the files, get the class labels
        print("Grabbing image labels")

        # If the labels have been passed in the **kwargs, then there is no
        # need to compute them; otherwise, execute the _get_labels method
        if 'y' in kwargs.keys():
            labels = kwargs['y']
        else:
            labels = self._get_labels(img_files)

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
        model = multi_gpu_model(model=model, gpus=self._ngpu)
        return model

    # @staticmethod
    # def _convert_img(img: Image.Image) -> Image.Image:
    #     """Converts an image from gray-scale to RGB if necessary
    #     """
    #     return img.convert("RGB")

    @staticmethod
    def _resize_img(img: Image.Image, new_shape: tuple) -> Image.Image:
        """
        Re-sizes the image to the desired shape
        """
        # If the image is already at the desired shape then there's nothing
        # that we need to do
        if (img.height, img.width) == new_shape:
            return img
        else:
            return img.resize(new_shape)

    def _get_imgs(self, img_files: np.ndarray) -> np.ndarray:
        """Reads in the images from img_files
        """
        # Read in the subset of images
        with Parallel(n_jobs=1) as p:
            imgs = p(delayed(Image.open)(file) for file in img_files)
            # imgs = p(delayed(self._convert_img)(img) for img in imgs)

            # Reshape the images
            new_shape = (self._height, self._width)
            imgs = p(delayed(self._resize_img)(img, new_shape)
                     for img in imgs)

            # Convert the image to numpy arrays
            imgs = p(delayed(np.array)(img) for img in imgs)
        return np.array(imgs)

    def _image_generator(self, img_files: list, steps: int) -> np.ndarray:
        """
        Defines the image generator object we will use to make our
        transformations
        """
        for i in range(steps):
            # Get the image arrays
            files = img_files[(self._batch_size * i):
                              (self._batch_size * (i+1))]
            imgs = self._get_imgs(files)

            # Check the image size
            print(imgs.shape)

            # Standardize the images
            yield imgs

    def transform(self, **kwargs):
        """Gets the transformed image data
        """

        # From the **kwargs, check if we already have the labels, if so, then
        # there is no need to get them from the _get_img_files method

        # Get the image files and the corresponding labels
        file_dict = self._get_img_files(**kwargs)

        # Determine the number of steps it will take to get through
        # transforming all of the images
        if len(file_dict["img_files"]) % self._batch_size == 0:
            steps = len(file_dict["img_files"]) / self._batch_size
        else:
            steps = int(np.floor(len(file_dict["img_files"]) /
                                 self._batch_size)) + 1

        # Define our image generator
        img_gen = self._image_generator(
            img_files=file_dict["img_files"], steps=steps
        )

        # Define the model we will use to transform the data
        model = self._define_model()

        # Get all of the transformations
        print("Transforming image data")
        X = model.predict_generator(generator=img_gen, verbose=1, steps=steps)

        # Put the data in the expected format
        f = h5py.File(self._save_path, "w")
        f.create_dataset("X", data=X)
        f.create_dataset('y', data=file_dict['labels'])
        f.close()
        return None
