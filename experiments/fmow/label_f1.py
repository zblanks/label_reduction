import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import argparse
import os
from joblib import Parallel, delayed
import re


def get_prob_files(wd):
    """
    Gets the FC and HC probability matrix files
    """
    files = os.listdir(os.path.join(wd, 'proba_pred'))
    prob_files = []
    for file in files:
        if re.match(r'f|hc', file):
            prob_files.append(file)

    return prob_files


def extract_id(files):
    """
    Extracts the experimental ID from the probability files
    """
    n = len(files)
    exp_ids = [''] * n
    for (i, file) in enumerate(files):
        exp_ids[i] = re.sub(r'f_|hc_|.npy', '', file)

    return exp_ids


def adjust_method(method, group_algo):
    """
    Adjusts the method in the exp_df to more easily distinguish the approaches
    """
    if method == 'f':
        return 'FC'
    else:
        if group_algo == 'kmm':
            return 'KMC'
        elif group_algo == 'comm':
            return 'CD'
        elif group_algo == 'lp':
            return 'LP'
        elif group_algo == 'kmm-sc':
            return 'KMC-SC'
        else:
            return 'SC'


def compute_f1(y_test, prob_file):
    """
    Computes the F1 score for each label
    """

    Y_pred = np.load(prob_file)
    y_pred = Y_pred.argmax(axis=1)
    f1_vec = f1_score(y_test, y_pred, average=None)
    return f1_vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', type=str, nargs='?',
                        default='/pool001/zblanks/label_reduction_data/fmow')
    args = vars(parser.parse_args())

    wd = args['wd']

    exp_df = pd.read_csv(os.path.join(wd, 'experiment_settings.csv'))
    y_test = pd.read_csv(os.path.join(wd, 'test_labels.csv'), header=None)
    y_test = y_test.values.flatten()

    # The methods in exp_df need to be changed so that they can be more easily
    # separated when generating plots
    print('Adjusting method')
    exp_df['method'] = exp_df.apply(lambda row: adjust_method(row['method'],
                                                              row['group_algo']),
                                    axis=1)

    prob_files = get_prob_files(wd)
    exp_ids = extract_id(prob_files)
    df = exp_df.loc[exp_df['id'].isin(exp_ids), ['id', 'method', 'estimator']]

    prob_files = [os.path.join(wd, 'proba_pred', file) for file in prob_files]
    print('Computing F1 scores')
    with Parallel(n_jobs=-1, verbose=5) as p:
        f1_vecs = p(delayed(compute_f1)(y_test, file) for file in prob_files)
    F1_mat = np.array(f1_vecs)

    nlabels = len(np.unique(y_test))
    f1_col_names = ['f1_' + val for val in map(str, np.arange(nlabels))]
    f1_df = pd.DataFrame(data=F1_mat, columns=f1_col_names)
    f1_df['id'] = exp_ids
    print(f1_df.shape)
    print(df.shape)
    df = df.merge(f1_df, on='id')
    print(df.shape)
    df.to_csv(os.path.join(wd, 'f1_res.csv'), index=False)


if __name__ == '__main__':
    main()
