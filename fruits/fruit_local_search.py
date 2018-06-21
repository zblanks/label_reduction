import sys
sys.path.insert(0, '/home/zblanks/label_reduction')
from local_search import LocalSearch
from similarity_measure import SimilarityMeasure
import argparse
import os
import time
import pandas as pd
from multiprocessing import cpu_count
import numpy as np
from itertools import combinations


def compute_balance(z):
    """

    Parameters
    ----------
    z: array

    Returns
    -------
    float

    """
    # Compute the worst case balance
    n_class = z.shape[0]
    n_label = z.shape[1]
    max_label = n_class - n_label + 1
    worst_case = (max_label - 1) * (n_label - 1)

    # Compute the proposed solution's balance ratio
    combos = list(combinations(range(n_label), 2))
    tau_vals = np.empty(shape=(len(combos),))
    for (i, combo) in enumerate(combos):
        tau_vals[i] = np.abs(z[:, combo[0]].sum(axis=0) -
                             z[:, combo[1]].sum(axis=0))
    return tau_vals.sum() / worst_case


if __name__ == '__main__':
    # Set the seed so we have reproducible random search values
    np.random.seed(17)

    # Define the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', type=int)
    parser.add_argument('wd', nargs='?', type=str,
                        default='/pool001/zblanks/label_reduction_data')
    parser.add_argument('n_init', nargs='?', type=int, default=100)
    parser.add_argument('search_method', nargs='?', type=str,
                        default='inexact')
    parser.add_argument('mixing_vals', nargs='?', type=np.ndarray,
                        default=np.random.uniform(low=0, high=1, size=1000))
    parser.add_argument('label_vals', nargs='?', type=np.ndarray,
                        default=np.random.randint(low=5, high=30, size=1000,
                                                  dtype=int))
    parser.add_argument('metric', nargs='?', type=str, default='rbf')
    args = vars(parser.parse_args())

    # Get a DataFrame of our arguments which maps the task ID to the value
    # for the number of labels and mixing factor
    args_df = pd.DataFrame({'n_label': args['label_vals'],
                            'mixing_factor': args['mixing_vals']})

    # First we need to check if the auto-encoder has already been trained;
    # if it has; then we just need to read in the data that was encoded;
    # otherwise we have to train the model and use the data it encodes
    # to compute the similarity measure
    file = os.path.join(args['wd'], 'auto_encoder', 'encoded_data.csv')
    df = pd.read_csv(file)

    # Using the encoded data we now need to compute the similarity measure
    # we will use for the local search procedure
    combo_file = os.path.join(args['wd'], 'fruits_sim', 'combo_sim.csv')
    class_file = os.path.join(args['wd'], 'fruits_sim', 'class_sim.csv')
    if os.path.exists(combo_file) and os.path.exists(class_file):
        combo_sim = np.loadtxt(combo_file, delimiter=',')
        class_sim = np.loadtxt(class_file, delimiter=',')
    else:
        sm = SimilarityMeasure(df=df, metric=args['metric'])
        sm.run()
        combo_sim = sm.combo_sim
        class_sim = sm.class_sim
        np.savetxt(combo_file, X=combo_sim, delimiter=',')
        np.savetxt(class_file, X=class_sim, delimiter=',')

    # Finally we're going to perform local search and compute various metrics
    # of interest associated with the search
    task_id = args['task_id']
    ls = LocalSearch(n_label=args_df.loc[task_id, 'n_label'],
                     combo_sim=combo_sim, class_sim=class_sim,
                     n_init=args['n_init'],
                     mixing_factor=args_df.loc[task_id, 'mixing_factor'],
                     search_method=args['search_method'])
    start = time.time()
    ls.search()
    end = time.time()

    # Determine the total number of assignments made for the solution
    # so we can normalize the objective value
    n_assign = sum(ls.class_dict.values()) + sum(ls.combo_dict.values())

    # Compile a DataFrame that will allow us to store our results
    res_df = pd.DataFrame({'n_label': args_df.loc[task_id, 'n_label'],
                           'obj_val': ls.obj_val,
                           'obj_val_per_assign': ls.obj_val / n_assign,
                           'n_init': args['n_init'],
                           'converge_iter': ls.n_iter,
                           'time': (end - start),
                           'n_cpu': cpu_count(),
                           'mixing_factor': args_df.loc[task_id,
                                                        'mixing_factor'],
                           'balance_ratio': compute_balance(ls.label_map),
                           'search_method': args['search_method']},
                          index=[0])

    # Check if a .csv already exists; if not, we need to create it;
    # otherwise we need to append to the file
    file_path = os.path.join(args['wd'], 'fruits_res', 'local_search',
                             'local_search_res.csv')
    if os.path.exists(file_path):
        with open(file_path, 'a') as f:
            res_df.to_csv(f, header=False, index=False)
    else:
        res_df.to_csv(file_path, index=False)
