import numpy as np
from itertools import combinations, compress, repeat, tee, starmap
from scipy.special import comb
import pandas as pd
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

    max_iter: int
        The max number of iterations we will allow the algorithm to go
        until we we shut it off so that it does not search forever

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
                 max_try=1000, is_min=True, max_iter=1000):
        self._n_label = n_label
        self._combo_sim = combo_sim
        self._class_sim = class_sim
        self._random_seed = random_seed
        self._n_init = n_init
        self._mixing_factor = mixing_factor
        self._search_method = search_method
        self._max_try = max_try
        self._is_min = is_min
        self._max_iter = max_iter

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

            # Check to make sure we have a balanced solution
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

        # First check to ensure that each label has an assignment and then
        # check the balance constraints
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

    def _build_move_df(self, z):
        """Builds a DataFrame containing all possible ways we can search
        through the indexes

        Parameters
        ----------
        z: np.ndarray

        Returns
        -------
        pd.DataFrame

        """

        # First get the indexes which contain a combination
        combo_cols = np.where(z.sum(axis=0) > 1)[0]
        i_idx, j_idx = np.nonzero(z[:, combo_cols])
        j_idx = combo_cols[j_idx]

        # Generate all possible new columns we can move to
        new_cols = list(map(lambda col: np.setdiff1d(np.arange(self._n_label),
                                                     np.array([col])),
                            j_idx))
        new_cols = np.concatenate(new_cols)
        change_class = np.repeat(i_idx, repeats=(self._n_label - 1))
        old_cols = np.repeat(j_idx, repeats=(self._n_label - 1))
        return pd.DataFrame({'change_class': change_class,
                             'old_col': old_cols, 'new_col': new_cols})

    @staticmethod
    def _gen_neighbor(z, move_dict):
        """Generator which yields a complete neighbor of z

        Parameters
        ----------
        z: array

        move_dict: dict
            Dictionary containing the indexes for the change class, the old
            label, and the new label

        Returns
        -------
        np.ndarray
            z_neighbor

        """
        # Build the neighbor and return it
        z_neighbor = np.copy(z)
        z_neighbor[move_dict['change_class'], move_dict['old_col']] = 0
        z_neighbor[move_dict['change_class'], move_dict['new_col']] = 1
        return z_neighbor

    @staticmethod
    def _remove_bad_combos(combos, label):
        """Removes combinations that are not relevant for the swap that we
        made

        Parameters
        ----------
        combos: list
        label: int

        Returns
        -------
        List containing only the relevant combinations

        """
        return [elt for elt in combos if label in elt]

    def _update_dvs(self, z, z_best, combo_dict, move_dict):
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

        move_dict: dict
            Dictionary containing information on which class we moved, its
            old label, and the new new label

        Returns
        -------
        dict
            Dictionary containing the new class and combination dictionary

        """

        # Grab the class, old and new column
        class_change = move_dict['change_class']
        old_col = move_dict['old_col']
        new_col = move_dict['new_col']

        # First we need to change any (*, change_class) or (change_class, *)
        # combinations to zero, but everything else can be kept the same
        # since by construction we are only changing one class at a time
        new_combo_dict = combo_dict.copy()
        old_entries = np.where(z_best[:, old_col] > 0)[0]
        old_combos = list(combinations(old_entries, 2))
        old_combos = self._remove_bad_combos(old_combos, label=class_change)
        zero_vect = np.zeros(shape=(len(old_combos),), dtype=int)
        new_combo_dict.update(dict(zip(old_combos, zero_vect)))

        # Now we need to identify the entries in the column where we moved
        # one of the classes and then correspondingly update the combination
        # dictionary
        new_entries = np.where(z[:, new_col] > 0)[0]
        new_combos = list(combinations(new_entries, 2))
        new_combos = self._remove_bad_combos(new_combos, label=class_change)
        one_vect = np.ones(shape=(len(new_combos),), dtype=int)
        new_combo_dict.update(dict(zip(new_combos, one_vect)))

        # Finally we need to update the new lone class dictionary by
        # checking where the column sum equals 1
        class_dict = self._infer_lone_class(z)
        return {'class_dict': class_dict, 'combo_dict': new_combo_dict}

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

            # Check if we've hit the upper bound on the number of allowed
            # iterations
            if n_iter >= self._max_iter:
                break

            # Generate every possible move we can make and then permute the
            # rows of the DataFrame so that we search the space randomly
            move_df = self._build_move_df(z_best)
            move_df = move_df.iloc[np.random.permutation(len(move_df))]

            # Iterate over the search space
            for idx, row in move_df.iterrows():
                # Grab the relevant parameters to generate the neighbor
                move_dict = {'change_class': row['change_class'],
                             'old_col': row['old_col'],
                             'new_col': row['new_col']}

                # Generate the neighbor
                z_neighbor = self._gen_neighbor(z=z_best, move_dict=move_dict)

                # Check if the neighbor is feasible
                if not self._check_feasible_soln(z_neighbor):
                    continue

                # Infer the new DVs
                dv_dict = self._update_dvs(
                    z=z_neighbor, z_best=z_best, combo_dict=combo_dict,
                    move_dict=move_dict
                )
                new_class_dict = dv_dict['class_dict']
                new_combo_dict = dv_dict['combo_dict']

                # Compute the new objective value
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

    @staticmethod
    def _drop_rows(elt):
        """Helper function to help us determine which rows we need to drop
        from the DataFrame in a lazy manner so we don't use too much memory

        Parameters
        ----------
        elt: tuple

        Returns
        -------
        int or None
            Either a row that needs to be dropped or None

        """
        if elt[1] is False:
            return elt[0]

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
        dv_dict = self._infer_dvs(z_best)
        class_dict = dv_dict['class_dict']
        combo_dict = dv_dict['combo_dict']
        obj_val = self._compute_objective(class_dict, combo_dict)
        n_iter = 0

        # With exact local search we will consider the entire complete
        # neighborhood at each iteration and take the best solution
        # instead of just the first one that improves the solution
        while True:
            # Check if we've hit the upper bound of the number of allowed
            # iterations
            if n_iter >= self._max_iter:
                break

            # Get every possible move index
            move_df = self._build_move_df(z_best)
            move_list = map(lambda i: dict(move_df.loc[i, :]),
                            range(len(move_df)))
            move_list1, move_list2 = tee(move_list, 2)
            z_best_repeat = repeat(z_best)

            # Generate the entire z_neighborhood
            z_neighborhood = map(self._gen_neighbor, z_best_repeat, move_list1)
            neighborhood1, neighborhood2 = tee(z_neighborhood, 2)

            # Remove any infeasible solutions
            feasible_solns = map(self._check_feasible_soln, neighborhood1)
            solns1, solns2, solns3 = tee(feasible_solns, 3)
            z_neighborhood = compress(neighborhood2, solns1)
            move_list = compress(move_list2, solns2)

            # Go through every neighbor, infer the DV changes, and compute
            # the objective value
            combo_dict_repeat = repeat(combo_dict)
            fn_args = zip(z_neighborhood, z_best_repeat, combo_dict_repeat,
                          move_list)
            dv_dicts = starmap(self._update_dvs, fn_args)
            dv_dicts1, dv_dicts2 = tee(dv_dicts, 2)
            class_dicts = map(lambda elt: elt['class_dict'], dv_dicts1)
            combo_dicts = map(lambda elt: elt['combo_dict'], dv_dicts2)
            obj_vals = list(map(self._compute_objective, class_dicts,
                                combo_dicts))

            # If we have any sort of improvement; if we don't then we've
            # reached a local optimum
            if self._is_min:
                check = (min(obj_vals) < obj_val)
            else:
                check = (max(obj_vals) > obj_val)

            if check:
                if self._is_min:
                    obj_val = min(obj_vals)
                    best_soln_idx = int(np.argmin(obj_vals))
                else:
                    obj_val = max(obj_vals)
                    best_soln_idx = int(np.argmax(obj_vals))

                # We need to re-compute the best neighbor and the
                # DVs because we never fully put the neighborhood or the DV
                #  dictionaries into memory since the neighborhood can be
                # quite large
                feasible_solns = map(self._drop_rows,
                                     enumerate(solns3))
                feasible_solns = filter(None.__ne__, feasible_solns)
                move_df = move_df.drop(feasible_solns)
                move_df = pd.DataFrame(move_df).reset_index(drop=True)
                move_dict = dict(move_df.loc[best_soln_idx, :])
                z_old = z_best
                z_best = self._gen_neighbor(z_best, move_dict)
                dv_dict = self._update_dvs(
                    z=z_best, z_best=z_old, combo_dict=combo_dict,
                    move_dict=move_dict
                )
                combo_dict = dv_dict['combo_dict']
                class_dict = dv_dict['class_dict']
                n_iter += 1
            else:
                # Break we've found the local optimum
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
