from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.utils import multi_gpu_model
from glob import glob
import os
import numpy as np
from PIL import Image
import h5py
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

    def _get_labels(self, files: np.ndarray) -> np.ndarray:
        """
        Infers the labels from the provided image files
        """

        # We expect a file to have a format:
        # /path/to/file/data/label/filename.jpg and so we need to list the
        # directory to find the images because we assume it has the file
        # format shown below
        targets = os.listdir(self._data_path)

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
        labels = [os.path.basename(os.path.dirname(file)) for file in files]

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
        """Defines the CNN used to transform the data
        """
        if self._model_name == "xception":
            model = Xception(include_top=False, pooling="max")
        elif self._model_name == "densenet":
            model = DenseNet201(include_top=False, pooling="max")
        elif self._model_name == "inception":
            model = InceptionV3(include_top=False, pooling="max")
        elif self._model_name == "nasnet":
            model = NASNetLarge(include_top=False, pooling="max")
        else:
            model = InceptionResNetV2(include_top=False, pooling="max")

        # Sometimes we only have one GPU so Keras will automatically detect
        # this; otherwise we have to specify this setting
        if self._ngpu <= 1:
            return model
        else:
            return multi_gpu_model(model=model, gpus=self._ngpu)

    @staticmethod
    def _resize_img(img: Image.Image, new_shape: tuple) -> Image.Image:
        """
        Re-sizes the image to the desired shape
        """
        return img.resize(new_shape)

    @staticmethod
    def _get_imgs(img_files: np.ndarray) -> np.ndarray:
        """Reads in the images from img_files
        """
        # Read in the subset of images
        with Parallel(n_jobs=-1) as p:
            imgs = p(delayed(Image.open)(file) for file in img_files)
            # imgs = p(delayed(self._convert_img)(img) for img in imgs)

            # # Reshape the images
            # new_shape = (self._width, self._height)
            # imgs = p(delayed(self._resize_img)(img, new_shape)
            #          for img in imgs)

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

            # Account for the case where the batch size is one
            if self._batch_size == 1:
                imgs = imgs.reshape(1, imgs[0], imgs[1], imgs[2])

            # Standardize the images
            yield imgs

    def transform(self, **kwargs):
        """Gets the transformed image data
        """

        # It is possible that we already have the image data in a matrix and
        # thus all we have to do is define the model and then get the
        # predictions
        if 'X' in kwargs.keys():
            # Define the model to transform the data
            model = self._define_model()

            # Transform the image data
            print("Transforming image data")
            X = model.predict(kwargs['X'], verbose=1)

            # If X has been provided then we know y must have been given and
            # thus we'll have it in the expected format to save to disk
            file_dict = {"labels": kwargs['y']}

        else:
            # Get the image files and the corresponding labels
            file_dict = self._get_img_files(**kwargs)

            # Determine the number of steps it will take to get through
            # transforming all of the images
            if len(file_dict["img_files"]) % self._batch_size == 0:
                steps = int(len(file_dict["img_files"]) / self._batch_size)
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
            X = model.predict_generator(generator=img_gen, verbose=1,
                                        steps=steps)

        # Do reduce computation time and assist with down-stream classification
        # we will use LDA
        y = file_dict['labels'].flatten()
        nlabels = len(np.unique(y))

        # If the number of features is less than the labels then we cannot
        # perform LDA
        if nlabels >= X.shape[1]:
            X_new = np.copy(X)
        else:
            lda = LinearDiscriminantAnalysis(n_components=(nlabels - 1))
            X_new = lda.fit_transform(X, y)

        # Put the data in the expected format
        f = h5py.File(self._save_path, "w")
        f.create_dataset("X", data=X_new)
        f.create_dataset('y', data=y)
        f.close()
        return None
