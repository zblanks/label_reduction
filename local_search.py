import numpy as np
from itertools import combinations, compress
from scipy.special import comb
from multiprocessing import Pool


class LocalSearch(object):
    """Local Search method for our label reduction problem

    Parameters
    ----------
    n_label: int
        Number of labels to remap our classes to

    combo_sim : array, shape=(n_combo,)
        Similarity measure for all pairwise combinations

    class_sim : array, shape=(n_class,)
        Similarity measure for all lone classes

    random_seed : int, default: 17
        Defines the random seed for the local search

    n_init : int, default: 10
        Number of random restarts to use for the local search method

    mixing_factor: float
        Max percentage of the max number of classes that can be feasibly
        assigned to a given label to allow

    search_method: str
        Whether we are going to do exact or inexact local search

    Attributes
    ----------
    label_map : array, shape=(n_class, n_label)
        Best label map found through local search

    obj_val : float
        Objective value of the best label map found through local search

    n_iter : int
        Number of iterations for the local search to converge to local
        optima

    """

    def __init__(self, n_label, combo_sim, class_sim, random_seed=17,
                 n_init=10, mixing_factor=1/2, search_method='inexact'):
        self.__n_label = n_label
        self.__combo_sim = combo_sim
        self.__class_sim = class_sim
        self.__random_seed = random_seed
        self.__n_init = n_init
        self.__mixing_factor = mixing_factor
        self.__search_method = search_method

        # Infer the number of classes and labels from the provided initial
        # solution
        self.__n_class = self.__class_sim.shape[0]

        # Set the seed for the remainder of the procedure
        np.random.seed(self.__random_seed)

        # Infer the total number of pairwise combinations for our setup
        self.__n_combo = int(comb(self.__n_class, 2))

        # Initialize the attributes of interest
        self.label_map = np.empty(shape=(self.__n_class, self.__n_label))
        self.obj_val = 0.
        self.n_iter = 0

    def __check_balance(self, z):
        """Computes the balance value a proposed solution z

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        bool:
            Whether the solution meets the balance requirement

        """
        # Compute the balance value of the proposed solution
        label_combos = list(combinations(range(self.__n_label), 2))
        tau = np.zeros(shape=(len(label_combos),), dtype=int)
        for (i, combo) in enumerate(label_combos):
            tau[i] = np.abs(z[:, combo[0]].sum(axis=0) -
                            z[:, combo[1]].sum(axis=0))
        balance = tau.sum().astype(int)

        # Compute the worst case scenario balance
        m = self.__n_class - self.__n_label + 1
        w = (m - 1) * (self.__n_label - 1)

        # Check if the solution works
        if balance < (self.__mixing_factor * w):
            return True
        else:
            return False

    def __gen_feasible_soln(self):
        """Generates a feasible solution from the initial LP solution

        Returns
        -------
        array, shape=(n_class, n_label): start_soln
            The initial feasible solution used for the local search method

        """
        # Define our initial matrix
        start_soln = np.zeros(shape=(self.__n_class, self.__n_label))

        # Get a list of all the classes we need to assign to a label
        classes = np.arange(self.__n_class)

        # Next we need to randomly select n_label of those classes and
        # place in this the initial solution to meet our requirement that
        # we have a class in each label
        init_class_assign = np.random.choice(classes, size=self.__n_label,
                                             replace=False)
        start_soln[init_class_assign, np.arange(self.__n_label)] = 1

        # Now we need to remove the classes we've already assigned and then
        # generate a random assignment for the remaining classes
        classes = classes[~np.in1d(classes, init_class_assign)]
        while True:
            # Make a random assignment for the remaining classes
            random_assign = np.random.choice(np.arange(self.__n_label),
                                             size=len(classes))
            proposed_soln = np.copy(start_soln)
            proposed_soln[classes, random_assign] = 1

            # Check if the proposed solution meets the balance requirements
            if self.__check_balance(proposed_soln):
                start_soln = np.copy(proposed_soln)
                break
        return start_soln

    def __gen_feasible_soln_garbage_fn(self, _):
        """Garbage function which allows us to call __gen_feasible_soln with map

        Parameters
        ----------
        _

        Returns
        -------

        """
        return self.__gen_feasible_soln()

    def __check_feasible_soln(self, z):
        """Checks whether the provided label map is a feasible solution

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        bool
            Whether or not the provided solution is feasible

        """

        # First we have to check to make sure that each label has at least
        # one class assigned to it
        if (z.sum(axis=0) == 0).sum() >= 1:
            return False

        # Next we have to check if we meet the balance constraints
        if self.__check_balance(z):
            return True
        else:
            return False

    def __infer_lone_class(self, z):
        """Infers the lone class

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        Returns
        -------
        dict: class_dict

        """
        lone_cols = (z.sum(axis=0) == 1).astype(int).reshape(-1, 1)
        lone_cols = np.matmul(z, lone_cols).flatten().astype(int).tolist()
        return dict(zip(range(self.__n_class), lone_cols))

    def __infer_dvs(self, z):
        """Infers the relevant decision variables from the provided label
        map

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        Returns
        -------
        dict: class_dict
        dict: combo_dict

        """

        # To determine when a class is by itself we need to identify any
        # columns which have only one entry
        class_dict = self.__infer_lone_class(z)

        # Determine the combinations present our label map
        combos = list(combinations(range(self.__n_class), 2))
        w = np.zeros(shape=(self.__n_combo,), dtype=int)
        combo_cols = np.where(z.sum(axis=0) > 1)[0]

        for (i, combo) in enumerate(combos):
            # Sum the elements of the combinations by column
            combo_sum = z[combo[0], combo_cols] + z[combo[1], combo_cols]
            if 2 in combo_sum:
                w[i] = 1
        combo_dict = dict(zip(combos, w))
        return class_dict, combo_dict

    def __compute_objective(self, class_dict, combo_dict):
        """Computes the objective value for the provided label map

        Parameters
        ----------
        class_dict: dict
            Dictionary stating if classes are by themselves

        combo_dict: dict
            Dictionary stating if a particular combination is present in
            a solution

        Returns
        -------
        float
            Objective value for given label map

        """
        # Compute the lone class similarity
        lone_class = np.fromiter(class_dict.values(), dtype=float)
        lone_class_sim = (self.__class_sim * lone_class).sum()

        # Compute the value of the pairwise similarity
        combo_vars = np.fromiter(combo_dict.values(), dtype=float)
        pairwise_sim = (self.__combo_sim * combo_vars).sum()
        return pairwise_sim + lone_class_sim

    @staticmethod
    def __find_combo_cols_idxbo_cols_idx(z):
        """Finds the indexes which correspond to combination columns

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        array, shape=(n_entry,)
            i indexes which correspond to the combination columns

        array, shape=(n_entry,)
            j indexes which correspond to the combination columns

        """

        # Get the indexes
        combo_cols = np.where(z.sum(axis=0) > 1)[0]
        i, j = np.nonzero(z[:, combo_cols])
        j = combo_cols[j]
        return i, j

    def __gen_neighbor(self, z, i_idx, j_idx):
        """Generator which yields a complete neighbor of z

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        i_idx: array
            i indexes which correspond to the combination columns

        j_idx: array
            j indexes which correspond to the combination columns

        Returns
        -------
        dict:
            Dictionary containing the neighbor solution and (i, j) index
            which was moved and the original column index

        """
        for (i, val) in enumerate(i_idx):
            orig_j_idx = j_idx[i]
            for j in range(self.__n_label):
                if j == orig_j_idx:
                    continue
                else:
                    z_neighbor = np.copy(z)
                    z_neighbor[val, orig_j_idx] = 0
                    z_neighbor[val, j] = 1
                    yield {'z_neighbor': z_neighbor, 'change_col': j,
                           'old_col': orig_j_idx}

    def __update_dvs(self, z, z_best, combo_dict, change_col, old_col):
        """Updates the DV dictionaries in a smart way so we don't have
        to completely re-infer the DVs after using a new neighbor

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        z_best: array, shape=(n_class, n_label)
            Current best solution

        combo_dict: dict
            Dictionary stating if a particular combination is present in
            a solution

        change_col: int
            The column that the change_class was moved to

        old_col: int
            The orignal column that the class resided in

        Returns
        -------
        dict: class_dict
        dict: combo_dict

        """

        # First we need to change any (*, change_class) or (change_class, *)
        # combinations to zero, but everything else can be kept the same
        # since by construction we are only changing one class at a time
        new_combo_dict = combo_dict.copy()
        old_entries = np.argwhere(z_best[:, old_col] > 0).flatten()
        old_combos = list(combinations(old_entries, 2))
        zero_vect = np.zeros(shape=(len(old_combos),), dtype=int)
        new_combo_dict.update(dict(zip(old_combos, zero_vect)))

        # Now we need to identify the entries in the column where we moved
        # one of the classes and then correspondingly update the combination
        # dictionary
        new_entries = np.argwhere(z[:, change_col] > 0).flatten()
        new_combos = list(combinations(new_entries, 2))
        one_vect = np.ones(shape=(len(new_combos),), dtype=int)
        new_combo_dict.update(dict(zip(new_combos, one_vect)))

        # Finally we need to update the new lone class dictionary by
        # checking where the column sum equals 1
        class_dict = self.__infer_lone_class(z)
        return class_dict, new_combo_dict

    def __inexact_search(self, z):
        """Performs inexact local search

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        Returns
        -------
        dict
            Dictionary containing the best label map, objective value, and the
            number of iterations needed to converge to a local optima

        """
        # Compute the objective value for the starting solution
        z_best = np.copy(z)
        class_dict, combo_dict = self.__infer_dvs(z_best)
        obj_val = self.__compute_objective(class_dict, combo_dict)
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

            # Infer the columns that have combination indexes so we can
            # use this to generate neighbors
            i_idx, j_idx = self.__find_combo_cols_idxbo_cols_idx(z_best)

            # Define our complete neighborhood generator
            z_neighborhood = self.__gen_neighbor(z_best, i_idx, j_idx)
            for neighbor in z_neighborhood:
                # Check if the solution is feasible
                if not self.__check_feasible_soln(neighbor['z_neighbor']):
                    continue

                # Grab the neighbor and (i, j) change index
                z_neighbor = neighbor['z_neighbor']
                change_col = neighbor['change_col']
                old_col = neighbor['old_col']

                # Infer the DV changes and compute the new objective
                # value
                new_class_dict, new_combo_dict = self.__update_dvs(
                    z=z_neighbor, z_best=z_best, combo_dict=combo_dict,
                    change_col=change_col, old_col=old_col
                )

                new_obj_val = self.__compute_objective(new_class_dict,
                                                       new_combo_dict)
                if new_obj_val > obj_val:
                    z_best = np.copy(z_neighbor)
                    obj_val = new_obj_val
                    combo_dict = new_combo_dict.copy()
                    n_iter += 1
                    change_z = True
                    break
        return {'z_best': z_best, 'obj_val': obj_val, 'n_iter': n_iter}

    def __exact_search(self, z):
        """Performs exact local search where we consider the entire
        neighborhood to find the best local max

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        Returns
        -------
        dict
            Dictionary containing the best label map, objective value, and the
            number of iterations needed to converge to a local optima
        """
        # Compute the objective value for the starting solution
        z_best = np.copy(z)
        class_dict, combo_dict = self.__infer_dvs(z_best)
        obj_val = self.__compute_objective(class_dict, combo_dict)
        n_iter = 0

        # With exact local search we will consider the entire complete
        # neighborhood at each iteration and take the best solution
        # instead of just the first one that improves the solution
        while True:
            # Get potential indexes which we can use to make swaps
            i_idx, j_idx = self.__find_combo_cols_idxbo_cols_idx(z_best)

            # Generate the entire z_neighborhood
            z_neighborhood = list(self.__gen_neighbor(z_best, i_idx,
                                                      j_idx))

            # Remove any infeasible solutions
            feasible_solns = [False] * len(z_neighborhood)
            for (i, neighbor) in enumerate(z_neighborhood):
                feasible_solns[i] = self.__check_feasible_soln(
                    z=neighbor['z_neighbor']
                )
            z_neighborhood = list(compress(z_neighborhood, feasible_solns))

            # Go through every neighbor and compute the objective
            obj_vals = [0.] * len(z_neighborhood)
            class_dicts = [{}] * len(z_neighborhood)
            combo_dicts = [{}] * len(z_neighborhood)
            for (i, neighbor) in enumerate(z_neighborhood):
                z_neighbor = neighbor['z_neighbor']
                change_col = neighbor['change_col']
                old_col = neighbor['old_col']

                # Update the DVs based on the neighboring solution
                new_class_dict, new_combo_dict = self.__update_dvs(
                    z=z_neighbor, z_best=z_best, combo_dict=combo_dict,
                    change_col=change_col, old_col=old_col
                )

                # Compute the new objective value
                obj_vals[i] = self.__compute_objective(new_class_dict,
                                                       new_combo_dict)
                class_dicts[i] = new_class_dict
                combo_dicts[i] = new_combo_dict

            # Check if any of the solutions are better than the current best
            # solution
            if max(obj_vals) > obj_val:
                obj_val = max(obj_vals)
                best_soln_idx = int(np.argmax(obj_vals))
                z_best = z_neighborhood[best_soln_idx]['z_neighbor']
                combo_dict = combo_dicts[best_soln_idx].copy()
                n_iter += 1
            else:
                # Break we've found the local max
                break
        return {'z_best': z_best, 'obj_val': obj_val, 'n_iter': n_iter}

    def __single_search(self, z):
        """Performs local search to find the best label map given the
        provided starting point z

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        Returns
        -------
        dict
            Dictionary containing the best label map, objective value, and the
            number of iterations needed to converge to a local optima

        """
        if self.__search_method == 'inexact':
            soln_dict = self.__inexact_search(z)
        else:
            soln_dict = self.__exact_search(z)
        return soln_dict

    def search(self):
        """Performs local search n_init times and finds the best search
        given the provided starting point

        Returns
        -------
        object: self

        """

        # Assuming the provided solution was not already optimal, we need
        # to generate n_init starting solutions
        # Generate n_init starting solutions
        with Pool() as p:
            starting_solns = p.map(self.__gen_feasible_soln_garbage_fn,
                                   range(self.__n_init))

        # Using our starting solutions get the best solution for each
        # of them
        with Pool() as p:
            best_local_solns = p.map(self.__single_search, starting_solns)

        # Grab the objective values from each of the solutions, determine
        # which one is the best, and then return the solution and value
        # which correspond to it
        obj_vals = [0.] * len(best_local_solns)
        for i in range(len(best_local_solns)):
            obj_vals[i] = best_local_solns[i]['obj_val']

        # Try to get the best solution, but if we don't have a single
        # feasible solution pass junk data and inform the user
        try:
            best_soln = np.argmax(obj_vals)
            self.label_map = best_local_solns[best_soln]['z_best'].astype(int)
            self.obj_val = np.max(obj_vals)
            self.n_iter = best_local_solns[best_soln]['n_iter']
        except ValueError:
            print('Could not find feasible solution on iteration {}'.
                  format(self.__n_label))
            self.label_map = np.zeros(shape=(self.__n_class, self.__n_label))
            self.obj_val = 0.
            self.n_iter = 0
        return self
