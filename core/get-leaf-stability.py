from core.leaf_stability import get_leaf_stability
from sklearn.preprocessing import LabelBinarizer
from glob import glob
import pandas as pd
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str)
    args = vars(parser.parse_args())
    wd = args['wd']

    # Get all of the files that contain the full probability predictions
    hc_files = glob(os.path.join(wd, "proba_pred", "hc_*"))
    f_files = glob(os.path.join(wd, "proba_pred", "f_*"))
    files = hc_files + f_files

    # Get the indices that correspond to each unique label
    labels = pd.read_csv(os.path.join(wd, "test_labels.csv"), header=None)
    labels = labels.values
    uniq_labels = np.sort(np.unique(labels.flatten()))
    idx_list = [[]] * len(uniq_labels)
    for (i, label) in enumerate(uniq_labels):
        idx_list[i] = np.where(labels == label)[0]

    # One-hot encode the labels
    lb = LabelBinarizer()
    Y = lb.fit_transform(labels)

    # Compute the entropy for each of the probability matrices
    df = get_leaf_stability(files, idx_list, Y)

    # Save the entropy DataFrame to disk so we can do some analysis and
    # visualization in R
    df.to_csv(os.path.join(wd, "leaf_stability_res.csv"), index=False)


if __name__ == "__main__":
    main()
