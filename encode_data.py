from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import numpy as np
import os
import glob
from imageio import imread
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd


class EncodeData(object):
    """Encodes our data to be used for computing the similarity

    Parameters
    ----------
    data_path: str
        Path where the data is located

    save_path: str
        Path to save the model weights and training results

    n_gpu: int
        Number of GPUs used to train the auto-encoder

    n_epoch: int
        Number of epochs to train the auto-encoder

    patience: int
        Number of steps to wait until we stop training

    min_delta: float
        Min absolute change between epochs for early stopping

    batch_size: int
        Batch size to train the model

    random_seed: int
        Random seed for reproducibility

    lr: float
        Learning rate for the auto-encoder

    Attributes
    ----------
    encoded_data: array
        Encoded data from the auto-encoder

    model: Model
        Keras model object containing all of the training history

    train_history: array
        Array containing the training history of the auto-encoder

    """

    def __init__(self, data_path, save_path, n_gpu=2, n_epoch=50,
                 patience=2, min_delta=0.001, batch_size=64, random_seed=17,
                 lr=0.01):
        self.data_path = data_path
        self.save_path = save_path
        self.n_gpu = n_gpu
        self.n_epoch = n_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.lr = lr

        # Instantiate our encoded data objected
        self.encoded_data = np.empty(shape=(1, 1), dtype=float)
        self.train_history = np.empty(shape=(0, 2), dtype=float)

        # Define an placeholder Keras objects
        a = Input(shape=(1,))
        b = Dense(1,)(a)
        self.model = Model(inputs=a, outputs=b)
        self.parallel_model = Model(inputs=a, outputs=b)

        # Instantiate the image properties
        self.height = 0
        self.width = 0
        self.n_channel = 3

        # Define the number of steps we will need per epoch for our
        # training and validation data
        self.train_step = 0
        self.val_step = 0
        self.all_step = 0

        # Define a placeholder for our list of image files
        self.img_files = np.array([])
        self.train_files = np.array([])
        self.val_files = np.array([])

        # Define our random seed for the object
        np.random.seed(self.random_seed)

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

    @staticmethod
    def is_power2(x):
        """Checks if the provided number is a power of two

        Parameters
        ----------
        x: int

        Returns
        -------
        bool

        """
        return (x != 0) and ((x & (x - 1)) == 0)

    @staticmethod
    def nearest_power2(x):
        """Finds the nearest power of two that is strictly smaller than x

        Parameters
        ----------
        x: int

        Returns
        -------
        int

        """
        return 2**int(np.log2(x))

    def get_img_files(self):
        """Gets a list containing all of the image file paths

        Returns
        -------
        object: self

        """
        # We assume that data_path is the directory containing all of the
        # classes
        directories = os.listdir(self.data_path)
        self.img_files = [[]] * len(directories)
        for (i, directory) in enumerate(directories):
            self.img_files[i] = glob.glob(
                os.path.join(self.data_path, directory + '/*')
            )
        self.img_files = self.flatten_list(self.img_files)
        self.img_files = np.array(self.img_files)
        return self

    def split_img_files(self):
        """Splits our image files into training and validation set for
        the auto-encoder

        Returns
        -------
        object: self

        """
        # Use a 75-25 split for the files
        split_idx = np.random.choice([True, False], size=len(self.img_files),
                                     p=[0.75, 0.25])
        self.train_files = self.img_files[split_idx]
        self.val_files = self.img_files[~split_idx]
        return self

    def compute_img_shape(self):
        """Finds the appropriate image shape for all of the images in our
        (this assumes that all images in the directory have the same shape)

        Returns
        -------
        object: self

        """
        # Grab the first image in the list of image files, read it in, and
        # see what it's original shape is
        img = imread(self.img_files[0])

        # For our auto-encoder model to work, we assume that the height and
        # width of the image is the same and that they are a power of two;
        # thus we have to check those assumptions and if they are not
        # true we need to adjust the image to make them correct
        same_shape = (img.shape[0] == img.shape[1])
        height_power2 = self.is_power2(img.shape[0])
        width_power2 = self.is_power2(img.shape[1])
        if (not same_shape) or (not height_power2) or (not width_power2):
            # Find the closest power of two which ensures that our image
            # will be square and be a power of two
            self.height = self.nearest_power2(min(img.shape[0], img.shape[1]))
            self.width = self.nearest_power2(min(img.shape[0], img.shape[1]))
            self.n_channel = img.shape[2]
        return self

    def compute_train_val_step(self):
        """Computes the number of steps we need for each training and
        validation epoch

        Returns
        -------
        object: self

        """

        # Using this we can now compute the number of steps per training
        # and validation epoch
        if len(self.train_files) % self.batch_size == 0:
            self.train_step = len(self.train_files) / self.batch_size
        else:
            self.train_step = int(np.floor(len(self.train_files) /
                                           self.batch_size)) + 1

        if len(self.val_files) % self.batch_size == 0:
            self.val_step = len(self.val_files) / self.batch_size
        else:
            self.val_step = int(np.floor(len(self.val_files) /
                                         self.batch_size)) + 1

        if len(self.img_files) % self.batch_size == 0:
            self.all_step = len(self.img_files) / self.batch_size
        else:
            self.all_step = int(np.floor(len(self.img_files) /
                                         self.batch_size)) + 1
        return self

    @staticmethod
    def crop_image(img, new_shape):
        """Crops the provided image

        Parameters
        ----------
        img: array

        new_shape: tuple
            Desired new shape for the image

        Returns
        -------
        array
            Cropped image

        """
        # We assume that the new shape is different from the original shape
        # of the image, that the old shape is strictly greater than then
        # new shape and that both the new and old shape are square
        if img.shape == new_shape:
            return img

        # Crop the image
        offset = (img.shape[0] - new_shape[0]) // 2
        if img.shape[0] % 2 == 0:
            new_img = img[offset:(img.shape[0] - offset),
                          offset:(img.shape[1] - offset)]
            assert new_img.shape == new_shape, 'Cropped image does not match'
            return new_img
        else:
            new_img = img[offset:(img.shape[0] - offset - 1),
                          offset:(img.shape[1] - offset - 1)]
            assert new_img.shape == new_shape, 'Cropped image does not match'
            return new_img

    def get_images(self, files):
        """Reads in all of the images from the provided files

        Parameters
        ----------
        files: array

        Returns
        -------
        array:
            A (batch_size, height, width, n_channel) array of images

        """
        # Get in the images
        with Pool() as p:
            imgs = p.map(imread, files)

        with Pool() as p:
            imgs = p.starmap(
                self.crop_image,
                zip(imgs,
                    [(self.height, self.width,
                      self.n_channel)] * len(imgs))
            )
        imgs = np.array(imgs)
        imgs = imgs / 255
        return imgs

    def img_generator(self, data_source):
        """Defines our image generator which we will use to train our
        auto-encoder without putting all of the images into memory

        Parameters
        ----------
        data_source: str
            Whether we're working with the training, validation, or the
            entire set of data

        Returns
        -------
        array

        """
        # For all conditions we will grab the appropriate files, read in
        # the data, and yield the list as a generator so that we don't
        # have to put everything into memory
        if data_source == 'train':
            for i in range(self.train_step):
                files = self.train_files[(self.batch_size * i):
                                         (self.batch_size * (i+1))]
                yield self.get_images(files)

        elif data_source == 'validation':
            for i in range(self.val_step):
                files = self.val_files[(self.batch_size * i):
                                       (self.batch_size * i+1)]
                yield self.get_images(files)

        else:
            for i in range(self.all_step):
                files = self.img_files[(self.batch_size * i):
                                       (self.batch_size * i+1)]
                yield self.get_images(files)

    def define_model(self):
        """Defines the auto-encoder object; this architecture assumes
        that the images are square and the height and width are powers of
        two; if this is not true, then the architecture will not work

        Returns
        -------
        object: self

        """
        # Define the auto-encoder model
        img_input = Input(shape=(self.height, self.width,
                                 self.n_channel))
        x = Conv2D(filters=8, kernel_size=(3, 3), padding='same',
                   activation='relu')(img_input)
        x = MaxPooling2D(padding='same')(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = MaxPooling2D(padding='same')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = MaxPooling2D(padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = MaxPooling2D(padding='same')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = MaxPooling2D(padding='same', name='encoder')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=8, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=4, kernel_size=(3, 3), padding='same',
                   activation='relu')(x)
        x = UpSampling2D()(x)
        decoder = Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                         activation='sigmoid')(x)

        # Define the compiler
        self.model = Model(img_input, decoder)
        self.parallel_model = multi_gpu_model(model=self.model,
                                              gpus=self.n_gpu)
        self.parallel_model.compile(optimizer=Adam(lr=self.lr),
                                    loss='binary_crossentropy')
        return self

    def run_epoch(self):
        """Trains and validates the model for one epoch

        Returns
        -------
        dict:
            Dictionary containing the training and validation loss for the
            epoch

        """

        # Permute the list of training files so that the model does not
        # just learn the order
        self.train_files = np.random.permutation(self.train_files)

        # Define our train image generator object and a vector to hold
        # the loss values for each batch
        train_gen = self.img_generator(data_source='train')
        train_loss = np.empty(shape=(self.train_step,))

        # Go through each training step and update the model
        for i in tqdm(range(self.train_step)):
            x_train = next(train_gen)
            train_loss[i] = self.parallel_model.train_on_batch(x_train,
                                                               x_train)

        # Go through the validation set
        val_gen = self.img_generator(data_source='validation')
        val_loss = np.empty(shape=(self.val_step,))
        for i in range(self.val_step):
            x_val = next(val_gen)
            val_loss[i] = self.parallel_model.test_on_batch(x_val, x_val)

        # Get the average training and validation loss for the epoch
        loss_dict = {'train_loss': train_loss.mean(),
                     'val_loss': val_loss.mean()}
        return loss_dict

    def fit_model(self):
        """Fits the auto-encoder model

        Returns
        -------
        object: self

        """
        # If we don't improve for patience steps then we will the training
        # process
        best_loss = np.inf
        steps_since_improve = 0

        # Train the model for at most n_epoch iterations
        for i in range(self.n_epoch):
            # Check if we've hit the EarlyStopping condition
            if steps_since_improve == self.patience:
                break

            # Run our model for the particular epoch
            loss = self.run_epoch()
            loss_vect = np.array([loss['train_loss'], loss['val_loss']])
            loss_vect = loss_vect.reshape(-1, 2)
            self.train_history = np.append(self.train_history, loss_vect,
                                           axis=0)
            # Print the results from this epoch
            print('Train loss: {0:.4f}\n'
                  'Val loss: {0:.4f}'.format(loss['train_loss'],
                                             loss['val_loss']))

            # Check if we did any better this epoch within the tolerance
            if np.abs(loss['val_loss'] - best_loss) <= self.min_delta:
                steps_since_improve += 1
            else:
                # Update the loss value
                best_loss = loss['val_loss']
                steps_since_improve = 0

                # Save the model
                file = os.path.join(self.save_path, 'auto_encoder.h5')
                self.model.save(filepath=file)
        return self

    def get_encoded_data(self):
        """Gets the encoded data from the trained auto-encoder

        Returns
        -------
        object: self

        """

        # Define the model to get the encoded data
        encoder = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('encoder').output)
        encoder = multi_gpu_model(model=encoder, gpus=self.n_gpu)

        # Go through all of the data and generate the encoding
        data_gen = self.img_generator(data_source='all')
        data_list = [0.] * self.all_step
        for i in range(self.all_step):
            # Get the data
            img_data = next(data_gen)
            data_list[i] = encoder.predict_on_batch(img_data)

        # Collect all of the data
        self.encoded_data = np.array(data_list)
        return self

    def get_labels(self):
        """Gets the labels from the data files

        Returns
        -------
        list
            List of labels

        """
        # Get the list of directories
        directories = os.listdir(os.path.join(self.data_path))
        labels = [[]] * len(directories)
        for (i, directory) in enumerate(directories):
            labels[i] = [i] * len(os.listdir(os.path.join(
                self.data_path, directory)))
        labels = self.flatten_list(labels)
        return labels

    def encode(self):
        """Trains the auto-encoder and returns the encoded data in the
        format that is expected by the SimilarityMeasure class

        Returns
        -------
        object: self

        """

        # First we need to get the appropriate image files and get the
        # necessary information about our image data
        self.get_img_files()
        self.split_img_files()
        self.compute_img_shape()
        self.compute_train_val_step()

        # Now that we have the necessary files, if the auto-encoder has
        # already been saved then we can read it into disk; otherwise,
        # we are going to have to train the model
        if os.path.exists(os.path.join(self.save_path, 'auto_encoder.h5')):
            self.model = load_model(os.path.join(self.save_path,
                                                 'auto_encoder.h5'))
        else:
            self.define_model()
            self.fit_model()

        # Now that we have a model to work with, we need to get the encoded
        # data and the labels
        self.get_encoded_data()
        labels = self.get_labels()

        # Finally we need to put everything together to create our DataFrame
        # that the SimilarityMeasure object expects
        self.encoded_data = self.encoded_data.reshape(
            self.encoded_data.shape[0], -1)
        self.encoded_data = pd.DataFrame(self.encoded_data)
        self.encoded_data.loc[:, 'label'] = labels
        return self
