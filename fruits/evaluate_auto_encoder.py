import argparse
import os
from keras.models import load_model
from keras.utils import multi_gpu_model
import glob
from multiprocessing import Pool
import numpy as np
from PIL import Image


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


def get_img_files(path):
    """Gets a list of image files

    Parameters
    ----------
    path: str

    Returns
    -------
    list
        List of image files

    """
    directories = os.listdir(path)
    img_files = [[]] * len(directories)
    for (i, directory) in enumerate(directories):
        img_files[i] = glob.glob(os.path.join(path, directory + '/*'))
    return flatten_list(img_files)


def resize_image(image, new_shape):
    """Re-sizes the image to the desired shape

    Parameters
    ----------
    image: Image.Image
    new_shape: tuple

    Returns
    -------
    Image.Image

    """
    image.thumbnail(new_shape)
    return image


def get_images(files, height=64, width=64, n_channel=3):
    """Gets the images in an array to be used in the model

    Parameters
    ----------
    files: list
    height: int
    width: int
    n_channel: int

    Returns
    -------
    np.array

    """

    # Read in the images
    with Pool() as p:
        images = p.map(Image.open, files)

    # Resize the images to (height, width, n_channel)
    with Pool() as p:
        new_shape = (height, width, n_channel)
        images = p.starmap(resize_image, zip(images,
                                             [new_shape] * len(images)))

    # Convert the images to arrays
    with Pool() as p:
        images = p.map(np.array, images)
    images = np.array(images)
    images = images / 255
    return images


def convert_arr(img):
    """Converts an array to an Image.Image object

    Parameters
    ----------
    img: np.array

    Returns
    -------
    Image.Image

    """
    return Image.fromarray(np.uint8(img * 255))


def save_image(img, path):
    """Saves an Image.Image object to disk

    Parameters
    ----------
    img: Image.Image
    path: str

    Returns
    -------
    None

    """
    img.save(path)
    return None


def create_image_path(save_path, base_name):
    """Creates the image file path

    Parameters
    ----------
    save_path: str
    base_name: str

    Returns
    -------
    str

    """
    base_file = os.path.splitext(base_name)[0]
    file_path = os.path.join(save_path, base_file + '_decoded.jpg')
    return file_path


def save_decoded_images(img_files, save_path, images):
    """Saves the decoded images to disk so that we can see if they make
    any sense

    Parameters
    ----------
    img_files: list
    save_path: str
    images: np.array

    Returns
    -------
    None

    """
    # Get the base names of each of the files so that we know what we're
    # looking at
    with Pool() as p:
        file_base = p.map(os.path.basename, img_files)

    # Convert our images to Image.Image objects
    with Pool() as p:
        imgs = p.map(convert_arr, images)

    # Get the list of new image file names
    with Pool() as p:
        img_paths = p.starmap(create_image_path,
                              zip([save_path] * len(file_base), file_base))

    # Save all of the images to disk
    with Pool() as p:
        p.starmap(save_image, zip(imgs, img_paths))
    return None


if __name__ == '__main__':
    # Get the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', type=str)
    parser.add_argument('n_gpu', nargs='?', type=int, default=2)
    parser.add_argument('batch_size', nargs='?', type=int, default=2048)
    args = vars(parser.parse_args())

    # Get the trained Keras model
    file = os.path.join(args['wd'], 'auto_encoder', 'auto_encoder.h5')
    model = load_model(file)
    model = multi_gpu_model(model=model, gpus=args['n_gpu'])

    # Get all of the fruit image files
    data_path = os.path.join(args['wd'], 'fruits-360/train')
    image_files = get_img_files(data_path)

    # Using the image files, get our image data
    img_data = get_images(image_files)

    # Get our decoded image predictions from the model
    # decoded_imgs = model.predict(x=img_data, batch_size=args['batch_size'],
    #                              verbose=1)
    decoded_imgs = model.predict(x=img_data, batch_size=args['batch_size'],
                                 verbose=1)

    # Save the decoded images to disk
    decode_path = os.path.join(args['wd'], 'decoded_images')
    save_decoded_images(image_files, decode_path, decoded_imgs)
