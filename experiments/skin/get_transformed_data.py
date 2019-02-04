from core.transform_data import TransformData
import argparse
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import re


def clean_jpg(file: str):
    """
    Cleans the .jpg file string
    """
    return re.sub(r".jpg", "", file)


def get_diagnosis(df: pd.DataFrame, img_id: str):
    """
    Gets the diagnosis for the particular image ID
    """
    return df.loc[df['image_id'] == img_id, 'dx'].item()


def get_labels(df: pd.DataFrame, file_path: str):
    """
    Gets the class labels for the skin cancer images
    """

    # Get a full list of the image files
    files = os.listdir(file_path)

    # Go through each of the files and get their label from the provided
    # DataFrame
    with Parallel(n_jobs=-1, verbose=5) as p:
        # First strip the .jpg from the string so we just have the Image ID
        img_ids = p(delayed(clean_jpg)(file) for file in files)

        # Use the image IDs, get the labels
        y = p(delayed(get_diagnosis)(df, img_id) for img_id in img_ids)

    # Convert the diagnosis strings to integers
    y = np.array(y)
    uniq_diagnoses = np.unique(y)
    label_map = dict(zip(uniq_diagnoses, range(len(uniq_diagnoses))))
    return np.array([label_map[val] for val in y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/skin")
    parser.add_argument('--model', type=str, nargs='?', default='densenet')
    parser.add_argument('--ngpus', type=int, nargs='?', default=2)
    parser.add_argument('--batch_size', type=int, nargs='?', default=32)
    args = vars(parser.parse_args())

    # We know all of the images are (450 x 600 x 3), so we can hard-code this
    # fact
    img_size = (450, 600, 3)

    # Get the data and save paths
    datapath = os.path.join(args['wd'], 'data')
    savepath = os.path.join(args['wd'], 'data.h5')

    # We need to get the labels for the data and provide it to the
    # TransformData object
    df = pd.read_csv(os.path.join(args['wd'], 'HAM10000_metadata.csv'))
    y = get_labels(df, datapath)

    # Using the labels and the argument paths, transform the image data and
    # save it to disk
    transformer = TransformData(datapath, savepath, model_name=args['model'],
                                ngpu=args['ngpus'], img_shape=img_size,
                                batch_size=args['batch_size'])
    transformer.transform(y=y)


if __name__ == '__main__':
    main()
