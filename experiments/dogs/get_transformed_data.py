from core.transform_data import TransformData
import argparse
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from sklearn.decomposition import PCA
import h5py


def get_label(label_df: pd.DataFrame, file: str):
    """
    Get the label for one file
    """
    file_id = os.path.basename(file).replace('.jpg', '')
    return label_df.loc[label_df['id'] == file_id, 'breed'].item()


def get_labels(label_df: pd.DataFrame, files: list):
    """
    Gets the labels for the images
    """

    # Go through each file string and get its label from the DataFrame
    with Parallel(n_jobs=-1, verbose=5) as p:
        y = p(delayed(get_label)(label_df, file) for file in files)

    # Re-map the labels to a number from their string versions
    uniq_breeds = label_df.breed.unique()
    label_map = dict(zip(uniq_breeds, range(len(uniq_breeds))))
    return np.array([label_map[val] for val in y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/dogs")
    parser.add_argument('--model', type=str, nargs='?', default='nasnet')
    parser.add_argument("--ngpus", type=int, nargs="?", default=2)
    args = vars(parser.parse_args())

    # Get the file and the label vector
    files = glob(os.path.join(args['wd'], 'train/*.jpg'))
    label_df = pd.read_csv(os.path.join(args['wd'], 'labels.csv'))
    y = get_labels(label_df, files)

    # Transform the data
    datapath = os.path.join(args['wd'], 'train')
    savepath = os.path.join(args['wd'], 'data.h5')

    height = 375
    width = 500
    img_shape = (height, width, 3)
    transformer = TransformData(datapath, savepath, model_name=args['model'],
                                ngpu=args['ngpus'], img_shape=img_shape)

    transformer.transform(y=y)

    # Since the NASNetLarge model yields 4000 features, I want to reduce the
    # dimensionality to start with so that we don't have to spend so much
    # time computing the PCA each time
    f = h5py.File(savepath, 'r+')
    X = np.array(f['X'])
    del f['X']
    pca = PCA(n_components=300, random_state=17)
    X = pca.fit_transform(X)
    f.create_dataset('X', data=X)
    f.close()


if __name__ == '__main__':
    main()
