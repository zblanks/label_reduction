import numpy as np
from itertools import combinations, repeat
from joblib import Parallel, delayed
from numba import njit


def _gen_neighbor(z: np.ndarray, move_dict: dict) -> np.ndarray:
    """Creates a complete neighbor the label map
    """
    # Build the neighbor and return it
    z_neighbor = np.copy(z)
    z_neighbor[move_dict["change_class"], move_dict["old_col"]] = 0
    z_neighbor[move_dict["change_class"], move_dict["new_col"]] = 1
    return z_neighbor


def _remove_bad_combos(combos: list, label: int) -> list:
    """Removes combinations that are not relevant for the swap that we
    made
    """
    return [elt for elt in combos if label in elt]


def _compute_abs_col_diff(z: np.ndarray, combo_idx: tuple) -> int:
    """Computes the absolute difference between two columns
    """
    i, j = combo_idx
    return np.abs(z[:, i].sum(axis=0) - z[:, j].sum(axis=0))


def _check_balance(z: np.ndarray, mixing_factor: float) -> bool:
    """Computes the balance value a proposed solution z
    """

    # Compute the balance value of the proposed solution
    n_class, n_label = z.shape
    label_combos = combinations(range(n_label), 2)
    label_map = repeat(z)
    tau = np.array(list(map(_compute_abs_col_diff, label_map,
                            label_combos)))
    balance = tau.sum().astype(int)

    # Compute the worst case scenario balance
    m = n_class - n_label + 1
    w = (m - 1) * (n_label - 1)

    # Check if the solution works
    if balance < (mixing_factor * w):
        return True
    else:
        return False


def _fix_infeasible_soln(z: np.ndarray, rng: np.random.RandomState,
                         max_try: int, mixing_factor: float) -> np.ndarray:
    """Fixes an infeasible proposed solution and attempts to make it
    feasible or will stop after max_try attempts
    """

    attempts = 0
    soln = np.copy(z)
    while True:
        # Check to see if we"ve hit the attempt limit
        if attempts >= max_try:
            soln = np.zeros(shape=soln.shape)
            break
        else:
            # Find the largest column and grab a random entry from this
            # column
            max_col = np.argmax(soln.sum(axis=0))
            random_entry = rng.choice(
                np.nonzero(soln[:, max_col])[0], size=1
            )

            # Find the smallest column and assign this entry to that
            # column
            min_col = np.argmin(soln.sum(axis=0))
            soln[random_entry, max_col] = 0
            soln[random_entry, min_col] = 1

            # Check if our solution is feasible
            if _check_balance(soln, mixing_factor=mixing_factor):
                break
            else:
                attempts += 1
    return soln


def _gen_feasible_soln(rng: np.random.RandomState, n_class: int,
                       n_label: int, is_min: bool, max_try: int,
                       mixing_factor: float) -> np.ndarray:
    """Generates a feasible solution to star the local search
    """

    # Define our initial matrix
    start_soln = np.zeros(shape=(n_class, n_label), dtype=np.float32)

    # Get a list of all the classes we need to assign to a label
    classes = np.arange(n_class)

    # Next we need to randomly select n_label of those classes and
    # place in this the initial solution to meet our requirement that
    # we have a class in each label
    init_class_assign = rng.choice(classes, size=n_label, replace=False)
    start_soln[init_class_assign, np.arange(n_label)] = 1

    # Now we need to remove the classes we"ve already assigned and then
    # generate a random assignment for the remaining classes
    classes = classes[~np.in1d(classes, init_class_assign)]
    while True:
        # Make a random assignment for the remaining classes
        random_assign = rng.choice(np.arange(n_label), size=len(classes))
        proposed_soln = np.copy(start_soln)
        proposed_soln[classes, random_assign] = 1

        # If we"re minimizing we don"t have to check balance because
        # the algorithm will generate balanced solutions, but
        # we do if we"re maximizing
        if is_min:
            start_soln = np.copy(proposed_soln)
            break
        else:
            # Check to make sure we have a balanced solution
            if _check_balance(proposed_soln, mixing_factor=mixing_factor):
                start_soln = np.copy(proposed_soln)
                break
            else:
                start_soln = _fix_infeasible_soln(
                    z=proposed_soln, rng=rng, max_try=max_try,
                    mixing_factor=mixing_factor
                )
                break
    return start_soln.astype(np.float32)


@njit
def _infer_lone_class(z: np.ndarray) -> np.ndarray:
    """Infers the lone class label map
    """

    lone_cols = np.zeros(shape=(z.shape[1], 1), dtype=np.float32)
    lone_loc = np.where(z.sum(axis=0) == 1)[0]
    if len(lone_loc) == 0:
        return np.zeros(shape=(z.shape[0],), dtype=np.float32)
    else:
        one_vect = np.ones(shape=len(lone_loc))
        lone_cols[lone_loc] = one_vect
        # Use matrix multiplication with the label map and the lone columns
        # with the @ operator
        class_map = z @ lone_cols
        return class_map.flatten()


@njit
def _build_combos(itr: np.ndarray) -> list:
    """Builds all n choose 2 combinations from the list
    """

    n = len(itr)
    combos = [(0, 0)] * ((n * (n - 1)) // 2)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            combos[k] = (i, j)
            k += 1
    return combos


@njit
def _infer_dvs(z: np.ndarray) -> tuple:
    """Infers the relevant decision variables from the provided label
    map
    """

    # To determine when a class is by itself we need to identify any
    # columns which have only one entry
    n_class = z.shape[0]
    class_map = _infer_lone_class(z)

    # Determine the combinations present our label map
    n_combo = ((z.shape[0] * (z.shape[0] - 1)) // 2)
    class_combos = _build_combos(np.arange(n_class))
    combo_cols = np.where(z.sum(axis=0) > 1)[0]
    w = np.zeros(shape=(n_combo,), dtype=np.float32)
    for (i, combo) in enumerate(class_combos):
        for col in combo_cols:
            if z[combo[0], col] + z[combo[1], col] == 2:
                w[i] = 1.
    return class_map, w


@njit
def _build_new_cols(idx_arr: np.ndarray, n_label: int) -> np.ndarray:
    """Builds the new cols for the neighborhood DataFrame
    """
    new_cols = [0] * (len(idx_arr) * (n_label - 1))
    j = 0
    for idx in idx_arr:
        for i in range(n_label):
            if i == idx:
                continue
            else:
                new_cols[j] = i
                j += 1
    return np.array(new_cols).reshape(-1, 1)


@njit
def _repeat_arr(arr: list, repeats: int) -> np.ndarray:
    """Repeats a list to help build the neighborhood move array
    """

    repeat_arr = [0] * (len(arr) * repeats)
    j = 0
    for val in arr:
        for i in range(repeats):
            repeat_arr[j] = val
            j += 1
    return np.array(repeat_arr).reshape(-1, 1)


def _compute_objective(class_sim: np.ndarray,
                       combo_sim: np.ndarray,
                       class_map: np.ndarray,
                       combo_map: np.ndarray) -> np.ndarray:
    """Computes the objective value for the provided label maps
    """
    return np.dot(class_sim, class_map) + np.dot(combo_sim, combo_map)


def _compute_obj_change(old_combo_sim: np.ndarray,
                        new_combo_sim: np.ndarray,
                        old_class_sim: np.ndarray,
                        new_class_sim: np.ndarray) -> float:
    """Computes the change in the objective value
    """
    if len(old_class_sim) != 0:
        old_obj = old_combo_sim.sum() + old_class_sim.sum()
    else:
        old_obj = old_combo_sim.sum()

    if len(new_class_sim) != 0:
        new_obj = new_combo_sim.sum() + new_class_sim.sum()
    else:
        new_obj = new_combo_sim.sum()
    return new_obj - old_obj


@njit
def _build_move_arr(z: np.ndarray) -> np.ndarray:
    """Builds a an array containing all ways we can search
    through the local neighborhood
    """

    # First get the indexes which contain a combination
    n_label = z.shape[1]
    combo_cols = np.where(z.sum(axis=0) > 1)[0]
    i_idx, j_idx = np.nonzero(z[:, combo_cols])
    j_idx = combo_cols[j_idx]

    # Generate all possible new columns we can move to
    new_cols = _build_new_cols(j_idx, n_label)

    # Repeat the class and the old column we moved
    # from to have a consistent array size
    change_class = _repeat_arr(i_idx, repeats=(n_label - 1))
    old_cols = _repeat_arr(j_idx, repeats=(n_label - 1))
    return np.concatenate((change_class, new_cols, old_cols), axis=1)


@njit
def _build_change_combos(itr: np.ndarray, label: int) -> list:
    """Builds the list of combos that changed
    """
    change_combos = [(0, 0)] * len(itr)
    for i in range(len(itr)):
        if label < itr[i]:
            change_combos[i] = (label, itr[i])
        else:
            change_combos[i] = (itr[i], label)
    return change_combos


def _combo_changes(z: np.ndarray, label: int, col: int) -> list:
    """Finds the combos that were changed
    """
    entries = np.where(z[:, col] > 0)[0]
    entries = np.delete(entries, np.argwhere(entries == label))
    return _build_change_combos(entries, label)


def _find_assignment_changes(z: np.ndarray, z_best: np.ndarray,
                             move_dict: dict) -> list:
    """Finds the assignment changes that were made with the
    neighbor
    """
    # Find the old and new combos
    old_combos = _combo_changes(z_best, move_dict["change_class"],
                                move_dict["old_col"])
    new_combos = _combo_changes(z, move_dict["change_class"],
                                move_dict["new_col"])

    # Find the old and new lone classes
    old_lone_class = np.where(z_best.sum(axis=0) == 1)[0].tolist()
    new_lone_class = np.where(z.sum(axis=0) == 1)[0].tolist()
    return [old_combos, new_combos, old_lone_class, new_lone_class]


def _get_similarity(sim_dict: dict, change_idx: list) -> np.ndarray:
    """Gets the similarity values for the provided change
    values and the similarity dictionary
    """
    similarity_values = list(map(lambda idx: sim_dict[idx], change_idx))
    return np.array(similarity_values, dtype=np.float32)


def _get_change_similarity(combo_sim: dict, class_sim: dict,
                           change_list: list) -> dict:
    """Gets the similarity values given the assignment changes
    that were made
    """
    sim_dicts = [combo_sim, combo_sim, class_sim, class_sim]
    similarity_values = list(map(_get_similarity, sim_dicts, change_list))
    keys = ["old_combo_sim", "new_combo_sim", "old_class_sim",
            "new_class_sim"]
    similarity_dict = {}
    for (i, key) in enumerate(keys):
        similarity_dict[key] = similarity_values[i]
    return similarity_dict


def _update_dvs(z: np.ndarray, old_combos: list, new_combos: list,
                combo_map: np.ndarray, combo_idx: dict) -> tuple:
    """Updates the DV dictionaries in a smart way so we don't have
    to completely re-infer the DVs after using a new neighbor
    """

    # First we need to change any (*, change_class) or (change_class, *)
    # combinations to zero, but everything else can be kept the same
    # since by construction we are only changing one class at a time
    idx = list(map(lambda combo: combo_idx[combo], old_combos))
    zero_vect = np.zeros(shape=(len(old_combos),), dtype=np.float32)
    combo_map[idx] = zero_vect

    # Now we need to identify the entries in the column where we moved
    # one of the classes and then correspondingly update the combination
    # dictionary
    idx = list(map(lambda combo: combo_idx[combo], new_combos))
    one_vect = np.ones(shape=(len(new_combos),), dtype=np.float32)
    combo_map[idx] = one_vect

    # Finally we need to update the new lone class dictionary by
    # checking where the column sum equals 1
    class_map = _infer_lone_class(z)
    return class_map, combo_map


def _inexact_search(z: np.ndarray, rng: np.random.RandomState,
                    class_sim: dict, combo_sim: dict,
                    max_iter: int, is_min: bool, mixing_factor: float,
                    combo_idx: dict) -> dict:
    """Performs inexact local search for one iteration
    """

    # Compute the objective value for the starting solution
    z_best = np.copy(z)
    class_map, combo_map = _infer_dvs(z_best)
    obj_val = _compute_objective(np.array(list(class_sim.values())),
                                 np.array(list(combo_sim.values())),
                                 class_map=class_map, combo_map=combo_map)
    n_iter = 0
    change_z = True

    # Continuously loop until we reach the local search termination
    # condition
    while change_z:
        # Set the change_z to False and this will be updated if
        # we find a new z from the search; otherwise if we go through
        # an entire iteration without finding a better solution
        # we will exit out of the while loop
        change_z = False

        # Check if we"ve hit the upper bound on the number of allowed
        # iterations
        if n_iter >= max_iter:
            print("Reached max iterations; consider adding more")
            break

        # Generate every possible move we can make and then permute the
        # rows of the DataFrame so that we search the space randomly
        move_arr = _build_move_arr(z_best)
        permutation_idx = rng.permutation(np.arange(move_arr.shape[0]))
        move_arr = move_arr[permutation_idx, :]

        # Iterate over the search space
        for row in move_arr:
            # Grab the relevant parameters to generate the neighbor
            move_dict = {"change_class": row[0], "new_col": row[1],
                         "old_col": row[2]}

            # Generate the neighbor
            z_neighbor = _gen_neighbor(z=z_best, move_dict=move_dict)

            # Check the balance if we"re maximizing
            if not is_min:
                if not _check_balance(z_neighbor, mixing_factor=mixing_factor):
                    continue

            # Infer the assignment changes
            change_list = _find_assignment_changes(z_neighbor, z_best,
                                                   move_dict)

            # Get the similarity values for the assignment changes
            sim_dict = _get_change_similarity(combo_sim, class_sim,
                                              change_list)

            # Compute the change in the objective value
            obj_change = _compute_obj_change(
                sim_dict["old_combo_sim"], sim_dict["new_combo_sim"],
                sim_dict["old_class_sim"], sim_dict["new_class_sim"]
            )

            # Check if the new value is better depending on whether we're
            # minimizing or maximizing the objective
            if is_min:
                check = obj_change < 0
            else:
                check = obj_change > 0

            if check:
                z_best = np.copy(z_neighbor)
                obj_val += obj_change
                class_map, combo_map = _update_dvs(
                    z=z_best, old_combos=change_list[0],
                    new_combos=change_list[1], combo_map=combo_map,
                    combo_idx=combo_idx
                )
                n_iter += 1
                change_z = True
                break

    return {"z_best": z_best, "obj_val": obj_val, "n_iter": n_iter,
            "combo_map": combo_map, "class_map": class_map}


def _single_search(seed: int, n_label: int, n_class: int, max_try: int,
                   mixing_factor: float, is_min: bool,
                   class_sim: dict, combo_sim: dict,
                   max_iter: int, combo_idx: dict) -> dict:
    """Performs local search to find the best label map given the
    provided starting point z
    """

    # First define the RNG for this particular instance so we can have
    # reproducible results
    rng = np.random.RandomState(seed)

    # Generate a starting feasible solution
    z = _gen_feasible_soln(
        rng=rng, is_min=is_min, mixing_factor=mixing_factor, n_class=n_class,
        n_label=n_label, max_try=max_try
    )

    # Perform the local search for the starting solution
    return _inexact_search(
        z=z, rng=rng, class_sim=class_sim, combo_sim=combo_sim,
        max_iter=max_iter, is_min=is_min, mixing_factor=mixing_factor,
        combo_idx=combo_idx
    )


def search(n_label: int, combo_sim: dict, class_sim: dict,
           n_init=10, mixing_factor=0.25, max_try=1000, is_min=True,
           max_iter=10000) -> dict:
    """Performs local search n_init times and finds the best search
    given the provided starting point
    """

    # Perform local search by going through each of the instances and
    # providing a random seed for each instance so we get reproducible
    # results
    n_class = np.array(list(class_sim.values())).shape[0]
    n_combo = ((n_class * (n_class - 1)) // 2)
    combo_idx = dict(zip(combinations(range(n_class), 2),
                         range(n_combo)))
    with Parallel(n_jobs=-1, verbose=3) as p:
        best_local_solns = p(delayed(_single_search)(i, n_label, n_class,
                                                     max_try, mixing_factor,
                                                     is_min, class_sim,
                                                     combo_sim, max_iter,
                                                     combo_idx)
                             for i in range(n_init))

    # Grab the objective values from each of the solutions, determine
    # which one is the best, and then return the solution and value
    # which correspond to it
    obj_vals = list(map(lambda x: x["obj_val"], best_local_solns))

    # Try to get the best solution, but if we don"t have a single
    # feasible solution pass junk data
    try:
        if is_min:
            best_soln = int(np.argmin(obj_vals))
        else:
            best_soln = int(np.argmax(obj_vals))
    except ValueError:
        label_map = np.zeros(shape=(n_class, n_label))
        obj_val = np.nan
        n_iter = -1
        return {"label_map": label_map, "obj_val": obj_val, "n_iter": n_iter}

    # If we don"t get an error, get the relevant metrics
    label_map = best_local_solns[best_soln]["z_best"].astype(int)
    if is_min:
        obj_val = np.min(obj_vals)
    else:
        obj_val = np.max(obj_vals)
    n_iter = best_local_solns[best_soln]["n_iter"]
    class_map = best_local_solns[best_soln]["class_map"].astype(int)
    combo_map = best_local_solns[best_soln]["combo_map"].astype(int)
    return {"label_map": label_map, "obj_val": obj_val, "n_iter": n_iter,
            "class_map": class_map, "combo_map": combo_map}
