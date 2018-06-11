import os
import sys
import numpy as np
from multiprocessing import Pool
from itertools import combinations, compress


def argmax_approach(label_map):
    """Implements the arg max approach for generating feasible solutions
    by choosing the column which has the largest value for a given row

    Args:
        label_map (ndarray, shape=[n_class, n_labels]): LP optimal solution

    Returns:
        (ndarray, shape=[n_class, n_labels]): Feasible solution
    """

    # Get the highest value for each row
    high_vals = np.argmax(label_map, axis=1)

    # Create our new label map
    new_map = np.zeros(shape=label_map.shape, dtype=int)

    # Populate our new map with the appropriate values
    for i in range(new_map.shape[0]):
        new_map[i, high_vals[i]] = 1

    # Return the feasible solution
    return new_map


def prob_approach(label_map):
    """Implements the probabilistic approach to generating feasible solutions
    by using the row as the sampling distribution

    Args:
        label_map (ndarray, shape=[n_class, n_labels]): LP optimal solution

    Returns:
        (ndarray, shape=[n_class, n_labels]): Feasible solution
    """

    # Get our sampled values and a matrix to hold our results
    new_map = np.zeros(shape=label_map.shape, dtype=int)
    for i in range(new_map.shape[0]):

        # Sample the distribution and get the value for the given row
        prob_val = np.random.choice(a=np.arange(label_map.shape[1]), size=1,
                                    p=label_map[i, :])

        # Add the value to our new feasible solution matrix
        new_map[i, prob_val] = 1
    return new_map


def check_bad_solns(label_map):
    """Checks if a particular map is infeasible (we're looking for the
    occurrence of having a particular label have no values assigned to it

    Args:
        label_map (ndarray, shape=[n_class, n_labels]): Proposed solution

    Returns:
        bool: Whether the solution is feasible or not
    """

    # Check whether each column has at least one label in it
    if len(np.where(label_map.sum(axis=0) == 0)[0]) > 0:
        return False
    else:
        return True


def infer_dvs(label_map):
    """Infers the DVs used in the objective from the provided label map

    Args:
        label_map (ndarray, shape=[n_class, n_labels]): Proposed solution

    Returns:
        (ndarray, shape=[n_labels]) = Single label mapping DV
        (ndarray, shape=[n_comb, n_label]) = Combination label mapping DV
    """

    # If we provide an infeasible solution just return None (this is
    # when we want to compare our methods
    if label_map is None:
        return None

    # First determine when a label is by itself
    single_map = np.zeros(shape=(label_map.shape[0], 1), dtype=int)
    single_vals = np.where(label_map.sum(axis=0) == 1)[0]
    for val in single_vals:
        # Determine which original label got mapped to itself
        remap = np.where(label_map[:, val] == 1)[0]
        single_map[remap, 0] = 1

    # Determine the combinations that are present from our label map
    combos = list(combinations(range(label_map.shape[0]), 2))
    combo_map = np.zeros(shape=(len(combos), label_map.shape[1]), dtype=int)
    combo_values = np.where(label_map.sum(axis=0) > 1)[0]

    # Logically fill our combo_map according to the logic of the system
    for col in combo_values:
        for (i, combo) in enumerate(combos):
            if (label_map[combo[0], col] == 1) and \
                    (label_map[combo[1], col] == 1):
                combo_map[i, col] = 1

    # Return our maps
    return single_map, combo_map


def compute_objective(single_map, single_params, combo_map, combo_params):
    """Computes the objective value given the single and combination map
    that we have provided

    Args:
        single_map (ndarray, shape=[n_labels, 1]): Map indicating whether
        a label is by itself

        single_params (ndarray, shape=[n_labels, 1]): Parameters indicating
        the similarity for a single class

        combo_map (ndarray, shape=[n_combo, n_label]): Map indicating whether
        a combination is present

        combo_params (ndarray, shape=[n_combo, n_label]): Parameters
        indicating the similarity for a class combination

    Returns:
        float: Objective value
    """

    # Appropriately re-shape the combo parameters
    if combo_params.shape[1] == 1:
        combo_params = np.repeat(combo_params, repeats=combo_map.shape[1],
                                 axis=1)

    # Account for the situation when our logical DVs are None in which
    # case just return None
    if (single_map is None) and (combo_map is None):
        return np.nan

    # Compute the objective
    objective = (combo_params * combo_map).sum() + \
                np.dot(single_params.T, single_map)
    return float(objective)


def get_best_soln(wd, map_path, label_map, method, single_params,
                  combo_params, n_iter=None):
    """Gets the best feasible solution based on the provided label map

    Args:
        wd (str): Working directory to save objects to disk

        map_path (str): Path to desired location for the label maps

        label_map (ndarray, shape=[n_class, n_labels]): LP optimal solution

        method (str): Whether we're using the probability or arg max approach

        single_params (ndarray, shape=[n_labels, 1]): Parameters indicating
        the similarity for a single class

        combo_params (ndarray, shape=[n_combo, n_label]): Parameters
        indicating the similarity for a class combination

        n_iter (int): Number of random iterations to consider for the prob
        approach

    Returns:
        (ndarray, shape=[n_class, n_labels]): Final feasible solution
    """

    # Re-shape the combo parameters appropriately
    combo_params = np.repeat(combo_params, repeats=label_map.shape[1],
                             axis=1)

    # Check if we already have an integer solution; if this is the case
    # then we know that our solution must therefore be optimal and thus
    # we will just save this to disk
    if np.all((label_map == 0) | (label_map == 1)):
        # Save the map to disk
        fname = os.path.join(wd, map_path, 'final_map_orig_' +
                             str(label_map.shape[1]) + '.csv')
        np.savetxt(fname=fname, X=label_map, delimiter=',')

        # Infer the DVs and compute the objective
        single_map, combo_map = infer_dvs(label_map=label_map)
        obj_val = compute_objective(
            single_map=single_map, single_params=single_params,
            combo_map=combo_map, combo_params=combo_params
        )

        return label_map, obj_val, label_map.shape[1]

    # Get our new map
    if method == 'argmax':
        new_map = argmax_approach(label_map=label_map)

        # Check if our solution is feasible
        soln_check = check_bad_solns(label_map=new_map)
        if soln_check is True:
            print('Infeasible solution with k = {}; '
                  'try the probability approach. '
                  'Terminating program'.format(label_map.shape[1]))
            return None, np.nan, label_map.shape[1]

        # Compute the other logical variables so we can compute the
        # objective value
        single_map, combo_map = infer_dvs(label_map=new_map)
        obj_val = compute_objective(
            single_map=single_map, single_params=single_params,
            combo_map=combo_map, combo_params=combo_params
        )

        # Assuming we have a feasible solution, save it to disk so that
        # we can use it for later
        fname = os.path.join(wd, map_path, 'final_map_argmax_' +
                             str(new_map.shape[1]) + '.csv')
        np.savetxt(fname=fname, X=new_map, delimiter=',')
        return new_map, obj_val, label_map.shape[1]

    else:
        # Generate n_iter list of potential maps
        with Pool() as p:
            new_maps = p.map(prob_approach, [label_map] * n_iter)

        # Determine if we have any infeasible solutions and remove them
        # from our list
        with Pool() as p:
            bad_solns = p.map(check_bad_solns, new_maps)

        new_maps = list(compress(new_maps, bad_solns))
        n_maps = len(new_maps)

        # Infer our objective function decision variables for each map
        with Pool() as p:
            obj_dv = p.map(infer_dvs, new_maps)

        # Separate out the decision variables components
        single_map = [None] * n_maps
        combo_map = [None] * n_maps
        for (i, dv) in enumerate(obj_dv):
            single_map[i] = dv[0]
            combo_map[i] = dv[1]

        # Check that we have at least one solution
        if len(single_map) == 0 or len(combo_map) == 0:
            print('Found no feasible solutions with prob approach for k ='
                  ' {}'.format(label_map.shape[1]))
            return None, np.nan, label_map.shape[1]

        # Compute the objective value given the objective function decision
        # variables
        with Pool() as p:

            obj_vals = p.starmap(compute_objective,
                                 zip(single_map, [single_params] * n_maps,
                                     combo_map, [combo_params] * n_maps))

        # Determine the solution with the highest objective value and
        # save the corresponding map to disk
        best_obj_val = np.argmax(obj_vals)
        best_map = new_maps[best_obj_val]
        fname = os.path.join(wd, map_path, 'final_map_prob' +
                             str(best_map.shape[1]) + '.csv')
        np.savetxt(fname=fname, X=best_map, delimiter=',')
        return best_map, np.max(obj_vals), label_map.shape[1]


if __name__ == '__main__':
    # Get the script arguments
    wd = sys.argv[1]
    sim_path = sys.argv[2]
    map_path = sys.argv[3]
    which_method = sys.argv[4]
    k = sys.argv[5]
    n_iter = int(sys.argv[6])

    # Set the seed for reproducible results
    np.random.seed(17)

    # Get the LP label map and the model parameters
    path = os.path.join(wd, map_path, 'label_map_' + k + '.csv')
    lp_map = np.loadtxt(fname=path, delimiter=',')
    class_sim = np.loadtxt(fname=os.path.join(wd, sim_path, 'class_sim.csv'),
                           delimiter=',')
    combo_sim = np.loadtxt(fname=os.path.join(wd, sim_path, 'comb_sim.csv'),
                           delimiter=',')

    # Reshape our parameters appropriately
    class_sim = class_sim.reshape(-1, 1)
    combo_sim = combo_sim.reshape(-1, 1)

    # Generate the best feasible solution and save it to disk
    try:
        get_best_soln(wd=wd, map_path=map_path, label_map=lp_map,
                      method=which_method, n_iter=n_iter,
                      single_params=class_sim,
                      combo_params=combo_sim)
    except TypeError as e:
        print(e)
        print(k + ' did not work')
        sys.exit(1)
