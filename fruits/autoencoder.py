from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import os
import argparse
from multiprocessing import Pool
import numpy as np
import imageio
import glob
import pandas as pd


def flatten_list(x):
    """Flattens a list of lists

    Args:
        x (list): List of lists

    Returns:
        list: Flattened list
    """
    return [item for sublist in x for item in sublist]


def read_imgs(wd, is_train):
    """Reads in our images

    Args:
        wd (str): Working directory

    Returns:
        (ndarray, shape=[None, height, width, 3]): Matrix of image data
        is_train (bool): Whether we're working with the train or test data
    """

    # Get all of the directories for our training data
    directories = os.listdir(os.path.join(wd, 'fruits-360/train'))
    if is_train:
        files = [[]] * len(directories)
        for (i, directory) in enumerate(directories):
            files[i] = glob.glob(
                os.path.join(wd, 'fruits-360/train', directory + '/*')
            )
    else:
        files = [[]] * len(directories)
        for (i, directory) in enumerate(directories):
            files[i] = glob.glob(
                os.path.join(wd, 'fruits-360/test', directory + '/*')
            )

    # Flatten our list
    files = flatten_list(files)

    # Read in our image data
    with Pool() as p:
        imgs = p.map(imageio.imread, files)

    # Get the images into an array and then return our data
    imgs = np.array(imgs)
    return imgs


def define_model(height, width, n_gpu):
    """Defines the autoencoder object

    Args:
        height (int): Image height
        width (int): Image width
        n_gpu (int): Number of GPUs to train our model

    Returns:
        object: Keras model object
    """

    # Define our autoencoder model
    img_input = Input(shape=(height, width, 3))
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
               activation='relu')(img_input)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same',
               activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same',
               activation='relu')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same', name='encoder')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same',
               activation='relu')(encoded)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
               padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='valid',
               activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoded = Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid',
                     padding='same')(x)

    # Define the compiler
    autoencoder = Model(img_input, decoded)
    parallel_autoencoder = multi_gpu_model(model=autoencoder, gpus=n_gpu)
    parallel_autoencoder.compile(optimizer=Adam(lr=0.01),
                                 loss='mean_squared_error')
    return parallel_autoencoder


def train_model(wd, height, width, n_gpu, n_epoch, train_data, val_data):
    """Trains our autoencoder

    Args:
        wd (str): Working directory
        height (int): Image height
        width (int): Image width
        n_gpu (int): Number of GPUs used to train model
        n_epoch (int): Number of epochs to train model
        train_data (ndarray, shape=[None, height, width, 3]): Training data
        val_data (ndarray, shape=[None, height, width, 3]): Validation data

    Returns:
        object: Keras model object
    """

    # Get our Keras model
    model = define_model(height=height, width=width, n_gpu=n_gpu)

    # Define callbacks for our model
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(wd, 'autoencoder', 'autoencoder_weights.h5'),
        save_best_only=True, save_weights_only=True
    )

    logger = CSVLogger(filename=os.path.join(wd, 'autoencoder', 'error.csv'))

    early_stop = EarlyStopping(patience=2)

    # Train the model
    model.fit(
        x=train_data, y=train_data, epochs=n_epoch, verbose=2,
        callbacks=[checkpoint, logger, early_stop],
        validation_data=(val_data, val_data)
    )
    return model


def get_encoded_data(wd, height, width, n_gpu, n_epoch):
    """Gets the encoded data from the autoencoder object

    Args:
        wd (str): Working directory
        height (int): Image height
        width (int): Image width
        n_gpu (int): Number of GPUs used to train model
        n_epoch (int): Number of epochs to train model

    Returns:
        (ndarray, shape=?): Array of encoded data
    """

    # Get our image data and standardize it
    img_data = read_imgs(wd=wd, is_train=True)
    train_mean = img_data.mean()
    train_std = img_data.std()
    img_data = (img_data - train_mean) / train_std

    # Also grab the test data
    test_data = read_imgs(wd=wd, is_train=False)
    test_data = (test_data - train_mean) / train_std

    # Split our data so we can use validation
    np.random.seed(17)
    split_idx = np.random.choice(
        [True, False], size=img_data.shape[0], replace=True, p=[0.75, 0.25]
    )
    train_data = img_data[split_idx, :, :, :]
    val_data = img_data[np.logical_not(split_idx), :, :, :]

    # Check if the autoencoder already exists, if so, load the model;
    # we will train our autoencoder
    if os.path.exists(os.path.join(wd, 'autoencoder/autoencoder_weights.h5')):
        model = define_model(height=height, width=width, n_gpu=n_gpu)
        model.load_weights(os.path.join(wd, 'autoencoder',
                                        'autoencoder_weights.h5'))
    else:
        model = train_model(wd=wd, height=height, width=width, n_gpu=n_gpu,
                            n_epoch=n_epoch, train_data=train_data,
                            val_data=val_data)

    # Using the model we need to define a new Keras model which will allow
    # us to extract the transformed data
    encoder = Model(inputs=model.input,
                    outputs=model.layers[3].get_layer('encoder').output)

    # Get our encoded data
    train_encoded = encoder.predict(x=img_data)
    test_encoded = encoder.predict(x=test_data)
    return train_encoded, test_encoded


if __name__ == '__main__':
    # Get our scripts arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', help='Working directory for script', type=str)
    parser.add_argument('height', help='Image height', type=int)
    parser.add_argument('width', help='Image width', type=int)
    parser.add_argument('n_gpu', help='Number of GPUs', type=int)
    parser.add_argument('--n_epoch', help='Number of epochs', type=int,
                        required=False)
    args = vars(parser.parse_args())

    # Create a directory to hold our autoencoder object if it doesn't
    # already exist
    if not os.path.exists(os.path.join(args['wd'], 'autoencoder')):
        os.mkdir(os.path.join(args['wd'], 'autoencoder'))

    # Get our encoded data
    x_train, x_test = get_encoded_data(
        wd=args['wd'], height=args['height'], width=args['width'],
        n_gpu=args['n_gpu'], n_epoch=args['n_epoch']
    )
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Add our data to a pandas DataFrame
    train_df = pd.DataFrame(x_train)
    test_df = pd.DataFrame(x_test)

    # Get the original labels for our data
    label_directories = os.listdir(os.path.join(args['wd'],
                                                'fruits-360/train'))
    train_labels = [[]] * len(label_directories)
    test_labels = [[]] * len(label_directories)
    for (i, directory) in enumerate(label_directories):
        train_labels[i] = [i] * len(os.listdir(
            os.path.join(args['wd'], 'fruits-360/train', directory)
        ))

        test_labels[i] = [i] * len(os.listdir(
            os.path.join(args['wd'], 'fruits-360/test', directory)
        ))

    # Flatten our list of labels and add to the data
    train_labels = flatten_list(train_labels)
    test_labels = flatten_list(test_labels)
    train_df.loc[:, 'label'] = train_labels
    test_df.loc[:, 'label'] = test_labels

    # Save the data to disk
    train_df.to_csv(os.path.join(args['wd'], 'fruits_matrix',
                                 'train_encoded.csv'),
              index=False)

    test_df.to_csv(os.path.join(args['wd'], 'fruits_matrix',
                                'test_encoded.csv'),
                   index=False)
