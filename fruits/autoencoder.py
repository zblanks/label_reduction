from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import os
import argparse
import multiprocessing
import numpy as np


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
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
               activation='relu')(img_input)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
               activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same',
               activation='relu')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same', name='encoder')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
               activation='relu')(encoded)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
               activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    decoded = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid',
                     padding='same')(x)

    # Define the compiler
    autoencoder = Model(img_input, decoded)
    parallel_autoencoder = multi_gpu_model(model=autoencoder, gpus=n_gpu)
    parallel_autoencoder.compile(optimizer='adadelta',
                                 loss='binary_crossentropy')
    return parallel_autoencoder


def train_model(wd, height, width, n_gpu, n_epoch):
    """Trains our autoencoder

    Args:
        wd (str): Working directory
        height (int): Image height
        width (int): Image width
        n_gpu (int): Number of GPUs used to train model
        n_epoch (int): Number of epochs to train model

    Returns:
        object: Keras model object
    """

    # Define the image generator so we can scale our images on the fly
    train_datagen = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True,
        validation_split=0.25
    )

    # Define the training generator object
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(wd, 'fruits-360/train'),
        target_size=(height, width),
        seed=17
    )

    # Get our Keras model
    model = define_model(height=height, width=width, n_gpu=n_gpu)

    # Define callbacks for our model
    checkpoint = ModelCheckpoint(filepath=os.path.join(wd, 'autoencoder',
                                                       'autoencoder.h5'),
                                 save_best_only=True)
    logger = CSVLogger(filename=os.path.join(wd, 'autoencoder', 'error.csv'))

    # Train the model
    model.fit_generator(
        generator=train_generator,
        epochs=n_epoch,
        callbacks=[checkpoint, logger],
        use_multiprocessing=True,
        workers=multiprocessing.cpu_count()
    )


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

    # Check if the autoencoder already exists, if so, load the model;
    # we will train our autoencoder
    if os.path.exists(os.path.join(wd, 'autoencoder/autoencoder.h5')):
        model = load_model(filepath=os.path.join(wd, 'autoencoder',
                                                 'autoencoder.h5'))
    else:
        model = train_model(wd=wd, height=height, width=width, n_gpu=n_gpu,
                            n_epoch=n_epoch)

    # Using the model we need to define a new Keras model which will allow
    # us to extract the transformed data
    encoder = Model(inputs=model.input,
                    outputs=model.get_layer('encoder').output)

    # Define the image generator so we can scale our images on the fly
    predict_datagen = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True
    )

    # Define the training generator object
    predict_generator = predict_datagen.flow_from_directory(
        directory=os.path.join(wd, 'fruits-360/train'),
        target_size=(height, width),
        seed=17
    )

    # Get our encoded data
    encoded_data = encoder.predict_generator(
        generator=predict_generator, use_multiprocessing=True,
        workers=multiprocessing.cpu_count(), verbose=1
    )
    return encoded_data


if __name__ == '__main__':
    # Get our scripts arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', help='Working directory for script', type=str)
    parser.add_argument('height', help='Image height', type=int)
    parser.add_argument('width', help='Image width', type=int)
    parser.add_argument('n_epoch', help='Number of epochs', type=int)
    parser.add_argument('n_gpu', help='Number of GPUs', type=int)
    args = vars(parser.parse_args())

    # Create a directory to hold our autoencoder object if it doesn't
    # already exist
    if not os.path.exists(os.path.join(args['wd'], 'autoencoder')):
        os.mkdir(os.path.join(args['wd'], 'autoencoder'))

    # Get our encoded data
    data = get_encoded_data(
        wd=args['wd'], height=args['height'], width=args['width'],
        n_gpu=args['n_gpu'], n_epoch=args['n_epoch']
    )

    # Save the data to disk
    np.save(file=os.path.join(args['wd'], 'autoencoder', 'encoded_data'),
            arr=data)
