import numpy as np
import imageio
import os
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import sys
import glob
import re
import pandas as pd


def read_flatten_img(file):
    """Reads and flattens the image

    Args:
        file (str): Image file

    Returns:
        ndarray (1, height * width): Vectorized image
    """

    # Read in the image
    img = imageio.imread(file)

    # Flatten and return the image
    return img.flatten().reshape(1, -1)


def get_label(file):
    """Grabs the class label from the file name

    Args:
        file (str): Image file name

    Returns:
        str: Label name
    """

    # Get the basename from the file
    label = os.path.basename(file)

    # Remove the number and file extension from the basename
    label = re.sub('(_[0-9])*([0-9]*)(.jpg$)', '', label)
    return label


if __name__ == '__main__':
    # Get the script arguments
    wd = sys.argv[1]

    # Get all of the training and testing files
    train_files = glob.glob(os.path.join(wd, 'train', '**', '*.jpg'))
    test_files = glob.glob(os.path.join(wd, 'test', '**', '*.jpg'))

    # Get image vectors for the training and test sets
    with Pool() as p:
        X_train = p.map(read_flatten_img, train_files)
        X_test = p.map(read_flatten_img, test_files)

    # Get the data into an array
    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)

    # Center and scale our data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get label strings
    with Pool() as p:
        train_labels = p.map(get_label, train_files)
        test_labels = p.map(get_label, test_files)

    # Re-map our labels to integers
    labels = os.listdir(os.path.join(wd, 'train'))
    label_map = dict(zip(labels, range(len(labels))))
    train_labels = np.vectorize(label_map.get)(train_labels)
    test_labels = np.vectorize(label_map.get)(test_labels)

    # Create a directory for our new matrix-based data
    if not os.path.exists(os.path.join(os.path.split(wd)[0], 'fruits_matrix')):
        os.mkdir(os.path.join(os.path.split(wd)[0], 'fruits_matrix'))

    # Save our data to disk in the new directory
    train = pd.DataFrame(X_train)
    train.loc[:, 'label'] = train_labels
    test = pd.DataFrame(X_test)
    test.loc[:, 'label'] = test_labels
    train.to_csv(os.path.join(os.path.split(wd)[0], 'fruits_matrix',
                              'train.csv'), index=False)
    test.to_csv(os.path.join(os.path.split(wd)[0], 'fruits_matrix',
                             'test.csv'), index=False)
