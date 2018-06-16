import sys
sys.path.insert(0, '/home/zblanks/label_reduction')
from old_scripts import gen_feasible_solns
import glob
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Get the script arguments
    wd = sys.argv[1]

    # Set the seed for reproducibility
    np.random.seed(17)

    # Get the LP solns
    label_map_paths = glob.glob(os.path.join(wd, 'fruit_maps/label_map*'),
                           recursive=True)
    n_maps = len(label_map_paths)
    label_maps = [None] * n_maps
    for (i, path) in enumerate(label_map_paths):
        label_maps[i] = np.loadtxt(path, delimiter=',')

    # Get the similarity measures
    class_sim = np.loadtxt(fname=os.path.join(wd, 'fruits_sim/class_sim.csv'),
                           delimiter=',')
    combo_sim = np.loadtxt(fname=os.path.join(wd, 'fruits_sim/comb_sim.csv'),
                           delimiter=',')

    # Reshape our parameters appropriately
    class_sim = class_sim.reshape(-1, 1)
    combo_sim = combo_sim.reshape(-1, 1)

    # Get our argmax solutions for all of the original LP solutions
    with Pool() as p:
        argmax_solns = p.starmap(
            gen_feasible_solns.get_best_soln,
            zip([wd] * n_maps, ['fruit_maps'] * n_maps, label_maps,
                ['argmax'] * n_maps, [class_sim] * n_maps,
                [combo_sim] * n_maps)
        )

    # Separate the argmax solutions
    argmax_obj_val = [None] * len(argmax_solns)
    argmax_k = [None] * len(argmax_solns)
    for i in range(len(argmax_solns)):
        argmax_obj_val[i] = argmax_solns[i][1]
        argmax_k[i] = argmax_solns[i][2]

    # Save the argmax objective value to disk
    fname = os.path.join(wd, 'fruits_res/method_comparison',
                         'argmax_obj_val.csv')
    df = pd.DataFrame({'obj_val': argmax_obj_val,
                       'n_label': argmax_k})
    df.to_csv(fname, index=False)

    # Get the probabilistic solutions for various number of iterations
    n_iter_vals = np.arange(start=100, stop=2100, step=100)
    prob_solns = [None] * (len(n_iter_vals) * len(label_maps))
    i = 0
    for val in n_iter_vals:
        for map in label_maps:
            tmp_prob_soln = gen_feasible_solns.get_best_soln(
                wd=wd, map_path='fruit_maps', label_map=map, method='prob',
                single_params=class_sim, combo_params=combo_sim,
                n_iter=val
            )

            # Add the solution to the list
            prob_solns[i] = tmp_prob_soln
            i += 1

    # Separate the components of our probabilistic approach
    prob_obj_val = [None] * len(prob_solns)
    prob_k = [None] * len(prob_solns)
    for i in range(len(prob_solns)):
        prob_obj_val[i] = prob_solns[i][1]
        prob_k[i] = prob_solns[i][2]

    # Save the result to disk
    df = pd.DataFrame({'obj_val': prob_obj_val,
                       'n_iter': np.tile(n_iter_vals, reps=len(label_maps)),
                       'n_label': prob_k})
    fname = os.path.join(wd, 'fruits_res/method_comparison',
                         'prob_obj_val.csv')
    df.to_csv(fname, index=False)
