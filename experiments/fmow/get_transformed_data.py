from os import path
from core.transform_data import TransformData
import argparse
import pandas as pd
import h5py
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    """
    Transforms the fMoW images into a X in R^{n x p} space to be used
    for our methods
    """

    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, nargs="?",
                        default="/pool001/zblanks")
    parser.add_argument('--model', type=str, nargs='?', default='nasnet')
    parser.add_argument('--ngpus', type=int, nargs='?', default=2)
    parser.add_argument('--batch_size', type=int, nargs='?', default=32)
    args = parser.parse_args()

    # Grab the command line arguments
    wd = args.wd
    model = args.model
    ngpus = args.ngpus
    batch_size = args.batch_size

    # Define the data transformation object
    datapath = path.join(wd, 'train_crop')
    savepath = path.join(wd, 'label_reduction_data/fmow/data.h5')
    transformer = TransformData(datapath, savepath, model, ngpus, batch_size)
    transformer.transform()

    # We are also going to generate the data which includes the meta-data
    # from the satellite photos
    meta_path = path.join(wd, 'side_info/metadata_train.csv')
    metadata = pd.read_csv(meta_path).values

    # Add the meta-data features and create a new .h5 file
    f = h5py.File(savepath, 'r')
    X = np.array(f['X'])
    y = np.array(f['y'])
    f.close()
    X_new = np.concatenate([X, metadata], axis=1)

    # Perform LDA on the new data so that it's in the expected form
    lda = LinearDiscriminantAnalysis(n_components=(len(np.unique(y)) - 1))
    X_new = lda.fit_transform(X_new, y)

    # Save the data to a new .h5 file
    h5_path = path.join(wd, 'label_reduction_data/fmow/data_meta.h5')
    f_new = h5py.File(h5_path, 'w')
    f_new.create_dataset('X', data=X_new)
    f_new.create_dataset('y', data=y)
    f_new.close()


if __name__ == "__main__":
    main()
