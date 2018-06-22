import numpy as np
from itertools import combinations, compress, repeat
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

    max_try: int
        Max number of times we will allow the algorithm to try to find
        a feasible solution before stopping it

    is_min: bool
        Whether we are minimizing or maximizing the objective function.
        This would change depending on what metric we are using

    Attributes
    ----------
    label_map : array
        Best label map found through local search

    obj_val : float
        Objective value of the best label map found through local search

    n_iter : int
        Number of iterations for the local search to converge to local
        optima

    combo_dict: dict
        Dictionary mapping the class combinations

    class_dict: dict
        Dictionary mapping the lone classes

    """

    def __init__(self, n_label, combo_sim, class_sim, random_seed=17,
                 n_init=10, mixing_factor=1/4, search_method='inexact',
                 max_try=1000, is_min=True):
        self._n_label = n_label
        self._combo_sim = combo_sim
        self._class_sim = class_sim
        self._random_seed = random_seed
        self._n_init = n_init
        self._mixing_factor = mixing_factor
        self._search_method = search_method
        self._max_try = max_try
        self._is_min = is_min

        # Infer the number of classes and labels from the provided initial
        # solution
        self._n_class = self._class_sim.shape[0]

        # Set the seed for the remainder of the procedure
        np.random.seed(self._random_seed)

        # Infer the total number of pairwise combinations for our setup
        self._n_combo = int(comb(self._n_class, 2))

        # Initialize the attributes of interest
        self.label_map = np.empty(shape=(self._n_class, self._n_label))
        self.obj_val = 0.
        self.n_iter = 0

        # Initialize the class and combination dictionaries
        self.class_dict = {}
        self.combo_dict = {}

    @staticmethod
    def _compute_abs_col_diff(z, combo_idx):
        """Computes the absolute difference between two columns

        Parameters
        ----------
        z: array

        combo_idx: tuple
            (i, j) index for the columns of interest

        Returns
        -------
        int

        """
        i, j = combo_idx
        return np.abs(z[:, i].sum(axis=0) - z[:, j].sum(axis=0))

    def _check_balance(self, z):
        """Computes the balance value a proposed solution z

        Parameters
        ----------
        z: array

        Returns
        -------
        bool:
            Whether the solution meets the balance requirement

        """
        # Compute the balance value of the proposed solution
        label_combos = combinations(range(self._n_label), 2)
        label_map = repeat(z)
        tau = np.array(list(map(self._compute_abs_col_diff, label_map,
                                label_combos)))
        balance = tau.sum().astype(int)

        # Compute the worst case scenario balance
        m = self._n_class - self._n_label + 1
        w = (m - 1) * (self._n_label - 1)

        # Check if the solution works
        if balance < (self._mixing_factor * w):
            return True
        else:
            return False

    def _fix_infeasible_soln(self, z):
        """Fixes an infeasible proposed solution and attempts to make it
        feasible or will stop after max_try attempts

        Parameters
        ----------
        z: array

        Returns
        -------
        array:
            Feasible solution

        """

        attempts = 0
        soln = np.copy(z)
        while True:
            # Check to see if we've hit the attempt limit
            if attempts >= self._max_try:
                soln = np.zeros(shape=soln.shape)
                break
            else:
                # Find the largest column and grab a random entry from this
                # column
                max_col = np.argmax(soln.sum(axis=0))
                random_entry = np.random.choice(
                    np.nonzero(soln[:, max_col])[0], size=1
                )

                # Find the smallest column and assign this entry to that
                # column
                min_col = np.argmin(soln.sum(axis=0))
                soln[random_entry, max_col] = 0
                soln[random_entry, min_col] = 1

                # Check if our solution is feasible
                if self._check_feasible_soln(soln):
                    break
                else:
                    attempts += 1
        return soln

    def _gen_feasible_soln(self):
        """Generates a feasible solution from the initial LP solution

        Returns
        -------
        array: start_soln
            The initial feasible solution used for the local search method

        """
        # Define our initial matrix
        start_soln = np.zeros(shape=(self._n_class, self._n_label))

        # Get a list of all the classes we need to assign to a label
        classes = np.arange(self._n_class)

        # Next we need to randomly select n_label of those classes and
        # place in this the initial solution to meet our requirement that
        # we have a class in each label
        init_class_assign = np.random.choice(classes, size=self._n_label,
                                             replace=False)
        start_soln[init_class_assign, np.arange(self._n_label)] = 1

        # Now we need to remove the classes we've already assigned and then
        # generate a random assignment for the remaining classes
        classes = classes[~np.in1d(classes, init_class_assign)]
        while True:
            # Make a random assignment for the remaining classes
            random_assign = np.random.choice(np.arange(self._n_label),
                                             size=len(classes))
            proposed_soln = np.copy(start_soln)
            proposed_soln[classes, random_assign] = 1

            # Check if the proposed solution meets the balance requirements
            # and we will only enforce the balance constraints for the
            # maximization version because when minimizing the objective
            # the solutions will typically want to be balanced to decrease
            # the number of combinations
            if not self._is_min:
                if self._check_balance(proposed_soln):
                    start_soln = np.copy(proposed_soln)
                    break
                else:
                    start_soln = self._fix_infeasible_soln(proposed_soln)
                    break
        return start_soln

    def _gen_feasible_soln_garbage_fn(self, _):
        """Garbage function which allows us to call _gen_feasible_soln with map

        Parameters
        ----------
        _

        Returns
        -------

        """
        return self._gen_feasible_soln()

    def _check_feasible_soln(self, z):
        """Checks whether the provided label map is a feasible solution

        Parameters
        ----------
        z: array

        Returns
        -------
        bool
            Whether or not the provided solution is feasible

        """

        # If we're minimizing the objective we will only enforce the
        # assignment constraints and not the balance constraint because
        # the solutions will want to be balanced, but we will enforce them
        # if we're maximizing the objective
        if self._is_min:
            if (z.sum(axis=0) == 0).sum() >= 1:
                return False
            else:
                return True
        else:
            if (z.sum(axis=0) == 0).sum() >= 1:
                return False

            if self._check_balance(z):
                return True
            else:
                return False

    def _infer_lone_class(self, z):
        """Infers the lone class

        Parameters
        ----------
        z : array

        Returns
        -------
        dict: class_dict

        """
        lone_cols = (z.sum(axis=0) == 1).astype(int).reshape(-1, 1)
        lone_cols = np.matmul(z, lone_cols).flatten().astype(int).tolist()
        return dict(zip(range(self._n_class), lone_cols))

    def _infer_dvs(self, z):
        """Infers the relevant decision variables from the provided label
        map

        Parameters
        ----------
        z : array

        Returns
        -------
        dict
            Dictionary containing both the class and combination dictionaries

        """

        # To determine when a class is by itself we need to identify any
        # columns which have only one entry
        class_dict = self._infer_lone_class(z)

        # Determine the combinations present our label map
        class_combos = list(combinations(range(self._n_class), 2))
        combo_cols = np.where(z.sum(axis=0) > 1)[0]
        w = [0] * len(class_combos)
        for (i, combo) in enumerate(class_combos):
            combo_sum = z[combo[0], combo_cols] + z[combo[1], combo_cols]
            if 2 in combo_sum:
                w[i] = 1
        combo_dict = dict(zip(class_combos, w))
        dv_dict = {'class_dict': class_dict, 'combo_dict': combo_dict}
        return dv_dict

    def _compute_objective(self, class_dict, combo_dict):
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
        lone_class_sim = (self._class_sim * lone_class).sum()

        # Compute the value of the pairwise similarity
        combo_vars = np.fromiter(combo_dict.values(), dtype=float)
        pairwise_sim = (self._combo_sim * combo_vars).sum()
        return pairwise_sim + lone_class_sim

    @staticmethod
    def _find_combo_cols_idx(z):
        """Finds the indexes which correspond to combination columns

        Parameters
        ----------
        z: array

        Returns
        -------
        array
            i indexes which correspond to the combination columns

        array
            j indexes which correspond to the combination columns

        """

        # Get the indexes
        combo_cols = np.where(z.sum(axis=0) > 1)[0]
        i, j = np.nonzero(z[:, combo_cols])
        j = combo_cols[j]
        return i, j

    def _gen_neighbor(self, z, i_idx, j_idx):
        """Generator which yields a complete neighbor of z

        Parameters
        ----------
        z: array

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
            for j in range(self._n_label):
                if j == orig_j_idx:
                    continue
                else:
                    z_neighbor = np.copy(z)
                    z_neighbor[val, orig_j_idx] = 0
                    z_neighbor[val, j] = 1
                    yield {'z_neighbor': z_neighbor, 'change_col': j,
                           'old_col': orig_j_idx}

    def _update_dvs(self, z, z_best, combo_dict, change_col, old_col):
        """Updates the DV dictionaries in a smart way so we don't have
        to completely re-infer the DVs after using a new neighbor

        Parameters
        ----------
        z : array

        z_best: array
            Current best solution

        combo_dict: dict
            Dictionary stating if a particular combination is present in
            a solution

        change_col: int
            The column that the change_class was moved to

        old_col: int
            The original column that the class resided in

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
        class_dict = self._infer_lone_class(z)
        return class_dict, new_combo_dict

    def _inexact_search(self, z):
        """Performs inexact local search

        Parameters
        ----------
        z : array

        Returns
        -------
        dict
            Dictionary containing the best label map, objective value, number
            of iterations to reach the local optima, the lone class map,
            and the combination class map

        """
        # Compute the objective value for the starting solution
        z_best = np.copy(z)
        dv_dict = self._infer_dvs(z_best)
        class_dict = dv_dict['class_dict']
        combo_dict = dv_dict['combo_dict']
        obj_val = self._compute_objective(class_dict, combo_dict)
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
            i_idx, j_idx = self._find_combo_cols_idx(z_best)

            # Define our complete neighborhood generator
            z_neighborhood = self._gen_neighbor(z_best, i_idx, j_idx)
            for neighbor in z_neighborhood:
                # Check if the solution is feasible
                if not self._check_feasible_soln(neighbor['z_neighbor']):
                    continue

                # Grab the neighbor and (i, j) change index
                z_neighbor = neighbor['z_neighbor']
                change_col = neighbor['change_col']
                old_col = neighbor['old_col']

                # Infer the DV changes and compute the new objective
                # value
                new_class_dict, new_combo_dict = self._update_dvs(
                    z=z_neighbor, z_best=z_best, combo_dict=combo_dict,
                    change_col=change_col, old_col=old_col
                )

                new_obj_val = self._compute_objective(new_class_dict,
                                                      new_combo_dict)

                # Check if the new value is better depending on whether we're
                # minimizing or maximizing the objective
                if self._is_min:
                    check = (new_obj_val < obj_val)
                else:
                    check = (new_obj_val > obj_val)

                if check:
                    z_best = np.copy(z_neighbor)
                    obj_val = new_obj_val
                    combo_dict = new_combo_dict.copy()
                    class_dict = new_class_dict.copy()
                    n_iter += 1
                    change_z = True
                    break
        return {'z_best': z_best, 'obj_val': obj_val, 'n_iter': n_iter,
                'combo_dict': combo_dict, 'class_dict': class_dict}

    def _exact_search(self, z):
        """Performs exact local search where we consider the entire
        neighborhood to find the best local max

        Parameters
        ----------
        z : array

        Returns
        -------
        dict
            Dictionary containing the best label map, objective value, number
            of iterations to reach the local optima, the lone class map,
            and the combination class map
        """
        # Compute the objective value for the starting solution
        z_best = np.copy(z)
        class_dict, combo_dict = self._infer_dvs(z_best)
        obj_val = self._compute_objective(class_dict, combo_dict)
        n_iter = 0

        # With exact local search we will consider the entire complete
        # neighborhood at each iteration and take the best solution
        # instead of just the first one that improves the solution
        while True:
            # Get potential indexes which we can use to make swaps
            i_idx, j_idx = self._find_combo_cols_idx(z_best)

            # Generate the entire z_neighborhood
            z_neighborhood = list(self._gen_neighbor(z_best, i_idx,
                                                     j_idx))

            # Remove any infeasible solutions
            feasible_solns = [False] * len(z_neighborhood)
            for (i, neighbor) in enumerate(z_neighborhood):
                feasible_solns[i] = self._check_feasible_soln(
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
                new_class_dict, new_combo_dict = self._update_dvs(
                    z=z_neighbor, z_best=z_best, combo_dict=combo_dict,
                    change_col=change_col, old_col=old_col
                )

                # Compute the new objective value
                obj_vals[i] = self._compute_objective(new_class_dict,
                                                      new_combo_dict)
                class_dicts[i] = new_class_dict
                combo_dicts[i] = new_combo_dict

            # Check if the new value is better depending on whether we're
            # minimizing or maximizing the objective
            if self._is_min:
                check = (min(obj_vals) < obj_val)
            else:
                check = (max(obj_vals) > obj_val)

            # Check if any of the solutions are better than the current best
            # solution
            if check:
                obj_val = max(obj_vals)
                best_soln_idx = int(np.argmax(obj_vals))
                z_best = z_neighborhood[best_soln_idx]['z_neighbor']
                combo_dict = combo_dicts[best_soln_idx].copy()
                class_dict = class_dict[best_soln_idx].copy()
                n_iter += 1
            else:
                # Break we've found the local max
                break
        return {'z_best': z_best, 'obj_val': obj_val, 'n_iter': n_iter,
                'combo_dict': combo_dict, 'class_dict': class_dict}

    def _single_search(self, z):
        """Performs local search to find the best label map given the
        provided starting point z

        Parameters
        ----------
        z : array

        Returns
        -------
        dict
            Dictionary containing the best label map, objective value, and the
            number of iterations needed to converge to a local optima

        """
        if self._search_method == 'inexact':
            soln_dict = self._inexact_search(z)
        else:
            soln_dict = self._exact_search(z)
        return soln_dict

    def search(self):
        """Performs local search n_init times and finds the best search
        given the provided starting point

        Returns
        -------
        object: self

        """

        # Generate n_init starting solutions or if that is not possible
        # give a junk answer that is not feasible
        with Pool() as p:
            starting_solns = p.map(self._gen_feasible_soln_garbage_fn,
                                   range(self._n_init))

        # Using our starting solutions get the best solution for each
        # of them
        with Pool() as p:
            best_local_solns = p.map(self._single_search, starting_solns)

        # Grab the objective values from each of the solutions, determine
        # which one is the best, and then return the solution and value
        # which correspond to it
        obj_vals = list(map(lambda x: x['obj_val'], best_local_solns))

        # Try to get the best solution, but if we don't have a single
        # feasible solution pass junk data
        try:
            best_soln = np.argmax(obj_vals)
        except ValueError:
            self.label_map = np.zeros(shape=(self._n_class, self._n_label))
            self.obj_val = np.nan
            self.n_iter = -1
            return self

        # If we don't get an error, get the relevant metrics
        self.label_map = best_local_solns[best_soln]['z_best'].astype(int)
        self.obj_val = np.max(obj_vals)
        self.n_iter = best_local_solns[best_soln]['n_iter']
        self.class_dict = best_local_solns[best_soln]['class_dict']
        self.combo_dict = best_local_solns[best_soln]['combo_dict']
        return self
