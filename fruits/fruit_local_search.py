import sys
sys.path.insert(0, '/home/zblanks/label_reduction')
from local_search import LocalSearch
import argparse
import glob
import os
import numpy as np
import time
import pandas as pd
from multiprocessing import cpu_count


if __name__ == '__main__':
    # Define the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', help='Working directory for script', type=str)
    parser.add_argument('n_init', help='Number of random initializations',
                        type=int)
    parser.add_argument('neighborhood_type', help='Type of neighborhood',
                        type=str)
    args = vars(parser.parse_args())

    # Get all of the initial LP solution files
    lp_soln_files = glob.glob(os.path.join(args['wd'], 'fruit_maps',
                                           'linear_*'))

    # Read in the data from the LP solutions
    lp_solns = [0.] * len(lp_soln_files)
    for (i, file) in enumerate(lp_soln_files):
        lp_solns[i] = np.loadtxt(file, delimiter=',')

    # Read in the similarity data
    combo_sim = np.loadtxt(os.path.join(args['wd'], 'fruits_sim',
                                        'comb_sim.csv'), delimiter=',')
    class_sim = np.loadtxt(os.path.join(args['wd'], 'fruits_sim',
                                        'class_sim.csv'), delimiter=',')

    # Perform local search for all of the provided solutions and let's
    # track how long each of them takes so that we have an idea of
    # how the algorithm is performing
    for (i, soln) in enumerate(lp_solns):
        ls = LocalSearch(init_soln=soln, combo_sim=combo_sim,
                         class_sim=class_sim, n_init=args['n_init'],
                         neighborhood_type=args['neighborhood_type'])

        # Time the search function to determine how it's performing
        start = time.time()
        ls.search()
        end = time.time()
        print('Iteration ' + str(i) + '; time: {0:.2f}'.format(end-start))

        # Grab the best map that was found and save this to disk
        best_z = ls.label_map_
        file_path = os.path.join(args['wd'], 'fruit_maps',
                                'local_map_' + str(i+2) + '.csv')
        np.savetxt(file_path, X=best_z, delimiter=',')

        # We will also save data on our how the search is doing by
        # looking at the objective value, number of restarts, and time to
        # finish computing
        res_df = pd.DataFrame({'n_label': best_z.shape[1],
                               'obj_val': ls.obj_val_,
                               'n_init': args['n_init'],
                               'converge_iter': ls.n_iter_,
                               'time': (end-start),
                               'n_cpu': cpu_count(),
                               'neighborhood_type': args['neighborhood_type']},
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
