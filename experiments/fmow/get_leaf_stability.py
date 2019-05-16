from core.leaf_stability import get_leaf_stability
from sklearn.preprocessing import OneHotEncoder
from glob import glob
import pandas as pd
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/fmow")
    args = vars(parser.parse_args())

    # Get all of the files that contain the full probability predictions
    hc_files = glob(os.path.join(args["wd"], "proba_pred", "hc_*"))
    f_files = glob(os.path.join(args["wd"], "proba_pred", "f_*"))
    files = hc_files + f_files

    # Get the indices that correspond to each unique label
    labels = pd.read_csv(os.path.join(args["wd"], "test_labels.csv"),
                         header=None)
    labels = labels.values.flatten()
    uniq_labels = np.sort(np.unique(labels))
    idx_list = [[]] * len(uniq_labels)
    for (i, label) in enumerate(uniq_labels):
        idx_list[i] = np.where(labels == label)[0]

    # One-hot encode the labels
    encoder = OneHotEncoder(n_values=len(uniq_labels))
    Y = encoder.fit_transform(labels.reshape(-1, 1))

    # Compute the entropy for each of the probability matrices
    df = get_leaf_stability(files, idx_list, Y)

    # Save the entropy DataFrame to disk so we can do some analysis and
    # visualization in R
    df.to_csv(os.path.join(args["wd"], "leaf_stability_res.csv"), index=False)


if __name__ == "__main__":
    main()
