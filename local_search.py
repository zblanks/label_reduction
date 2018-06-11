import numpy as np
from itertools import combinations
from scipy.special import comb
from multiprocessing import Pool


class LocalSearch(object):
    """Local Search method for our label reduction problem

    Parameters
    ----------
    init_soln : array, shape=(n_class, n_label)
        The LP initial solution used to start our local search method

    combo_sim : array, shape=(n_combo, 1)
        Similarity measure for all pairwise combinations

    class_sim : array, shape=(n_class, 1)
        Similarity measure for all lone classes

    correction_factor: array, shape=(combo_vals, 1)
        Correction factor for combinations of size k

    random_seed : int, default: 17
        Defines the random seed for the local search

    n_init : int, default: 100
        Number of random restarts to use for the local search method

    Attributes
    ----------
    label_map_ : array, shape=(n_class, n_label)
        Best label map found through local search

    obj_val_ : float
        Objective value of the best label map found through local search

    n_iter_ : int
        Number of iterations for the local search to converge to local
        optima

    """

    def __init__(self, init_soln, combo_sim, class_sim, correction_factor,
                 random_seed=17, n_init=10, neighborhood_type='complete'):
        self.init_soln = init_soln
        self.combo_sim = combo_sim
        self.class_sim = class_sim
        self.correction_factor = correction_factor
        self.random_seed = random_seed
        self.n_init = n_init
        self.neighborhood_type = neighborhood_type

        # Reshape the similarity measures
        self.combo_sim = self.combo_sim.reshape(-1, 1)
        self.class_sim = self.class_sim.reshape(-1, 1)

        # Infer the number of classes and labels from the provided initial
        # solution
        self.n_class, self.n_label = self.init_soln.shape

        # Set the seed for the remainder of the procedure
        np.random.seed(self.random_seed)

        # Infer the total number of pairwise combinations for our setup
        self.n_combo = int(comb(self.n_class, 2))

        # Initialize the attributes of interest
        self.label_map_ = np.empty(shape=(self.n_class, self.n_label))
        self.obj_val_ = 0.
        self.n_iter_ = 0

        # Initialize the list of dominated solutions to help speed up
        # computation time by not considering previously inferior solutions
        self.dominated_solns = []

    def gen_feasible_soln(self):
        """Generates a feasible solution from the initial LP solution

        Returns
        -------
        array, shape=(n_class, n_label): init_feasible_soln
            The initial feasible solution used for the local search method

        """

        # Define an empty array for our new initial feasible solution
        init_feasible_soln = np.zeros(shape=(self.n_class, self.n_label),
                                      dtype=int)

        # Sample from the LP and generate a feasible solution
        for i in range(self.n_class):
            prob_val = np.random.choice(a=np.arange(self.n_label), size=1,
                                        p=self.init_soln[i, :])
            init_feasible_soln[i, prob_val] = 1
        return init_feasible_soln

    def gen_feasible_soln_garbage_fn(self, _):
        """Garbage function which allows us to call gen_feasible_soln with map

        Parameters
        ----------
        _

        Returns
        -------

        """
        return self.gen_feasible_soln()

    @staticmethod
    def check_feasible_soln(z):
        """Checks whether the provided label map is a feasible solution

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        bool
            Whether or not the provided solution is feasible

        """

        # We only have to check if each label has more than one class in
        # it because our method producing initial feasible solutions will
        # automatically satisfy the property that each label has
        # only one class in it
        if len(np.where(z.sum(axis=0) == 0)[0]) == 0:
            return True
        else:
            return False

    def check_init_soln_is_opt(self):
        """Checks if the provided initial solution is optimal by seeing
        if all the array has all integer solutions

        Returns
        -------
        bool
            Determines if the LP solution is optimal or not

        """

        # Check if we have any entries in self.init_soln which are not
        # either 0 or 1
        if np.isin(self.init_soln, [0., 1.]).sum() < self.n_class*self.n_label:
            return False
        else:
            return True

    def build_combo_set(self):
        """Builds our combination set (including 0)

        Returns
        -------

        array, shape=(combo_vals,): combo_set
            Array containing all possible combination values for a particular
            label (to include 0)

        """

        i_vals = np.arange(self.n_class - self.n_label + 1)
        combo_set = np.empty(shape=(len(i_vals)), dtype=int)
        for (i, val) in enumerate(i_vals):
            combo_set[i] = int(comb(self.n_class - self.n_label + 1 - val, 2))
        return combo_set

    def infer_dvs(self, z):
        """Infers the relevant decision variables from the provided label
        map

        Parameters
        ----------
        z : array, shape=(n_class, n_label)

        Returns
        -------
        array, shape=(n_class, 1): y
            DV indicating that a particular class is by itself

        array, shape=(n_combo, n_label): w
            DV indicating that a particular combination is present in label
            j

        array, shape=(combo_vals, n_label): x
            DV indicating a combination of size k (including 0) is present
            in label j

        """

        # To determine when a class is by itself we need to identify any
        # columns which have only one entry
        x = np.zeros(shape=(self.n_label, 1))
        lone_cols = np.where(z.sum(axis=0) == 1)[0]
        x[lone_cols] = 1
        y = np.matmul(z, x)

        # Determine the combinations present our label map
        combos = combinations(range(self.n_class), 2)
        w = np.zeros(shape=(self.n_combo, self.n_label))
        combo_cols = np.where(z.sum(axis=0) > 1)[0]

        # TODO: Look for way to vectorize this operation
        for col in combo_cols:
            for (i, combo) in enumerate(combos):
                if (z[combo[0], col] == 1) and (z[combo[1], col] == 1):
                    w[i, col] = 1

        # Infer x for our correction factor
        combo_set = self.build_combo_set()
        combo_set = np.sort(combo_set)
        combo_dict = dict(zip(combo_set, range(len(combo_set))))
        delta = np.zeros(shape=(len(combo_set), self.n_label))
        combo_sum = w.sum(axis=0).astype(int)
        for (j, sum_val) in enumerate(combo_sum):
            delta[combo_dict[sum_val], j] = 1

        # Return the inferred maps
        return y, w, delta

    def compute_objective(self, z):
        """Computes the objective value for the provided label map

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        float
            Objective value for given label map

        """

        # First we need to infer the relevant decision variables from
        # the provided label map
        y, w, delta = self.infer_dvs(z)

        # Compute the lone class similarity
        lone_class_sim = float(np.matmul(self.class_sim.T, y))

        # Compute the value of the pairwise similarity
        combo_entries = w.sum(axis=0)
        combo_sim = self.combo_sim*w
        combo_sum = combo_sim.sum(axis=0)
        no_combos = combo_entries != 0
        pairwise_sim = np.zeros_like(combo_sum)
        np.place(pairwise_sim, no_combos,
                 combo_sum[no_combos]/combo_entries[no_combos])
        pairwise_sim = pairwise_sim.sum()

        # Compute the correction factor
        correction = self.correction_factor * delta
        correction = correction.sum()

        # Return the sum of the two components
        return pairwise_sim + lone_class_sim - correction

    @staticmethod
    def find_combo_cols_idx(z):
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

    def remove_infeasible_solutions(self, soln_list):
        """Removes any infeasible solutions from a list of provided solutions

        Parameters
        ----------
        soln_list: list
            List of potential solutions generated in the N(z)

        Returns
        -------
        list: list of feasible solutions

        """

        # Remove any infeasible solutions
        feasible_solns = []
        for (i, soln) in enumerate(soln_list):
            if self.check_feasible_soln(soln):
                feasible_solns.append(i)
        return list(np.array(soln_list)[feasible_solns])

    def gen_stochastic_neighborhood(self, z):
        """Generates a stochastic neighborhood of z

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        array: neighborhood_list

        """

        # First we need to select the (i, j) entry which we will use
        # to create the N(z) by determining if (i, j) is in a column with
        # more than one entry and then randomly picking from our list of
        # options
        i, j = self.find_combo_cols_idx(z)
        idx_choice = np.random.choice(np.arange(len(i)), size=1)
        i_choice, j_choice = i[idx_choice], j[idx_choice]

        # Now we need to generate the neighborhood by changing the value of
        # j for our (i, j) index to include all entries of j' = 1, ..., L
        # where j != j'
        neighborhood_list = []
        for j in range(self.n_label):
            if j == j_choice:
                continue
            else:
                new_map = np.copy(z)
                new_map[i_choice, j_choice] = 0
                new_map[i_choice, j] = 1
                neighborhood_list.append(new_map)
        return np.array(neighborhood_list)

    def gen_complete_neighborhood(self, z):
        """Generates a complete neighborhood by grabbing all indexes which
        are in combination columns and adjusting the column entry for
        all of them

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        array: neighborhood_list

        """

        # First get all of the indexes which correspond to our column entries
        i_idx, j_idx = self.find_combo_cols_idx(z)

        # Next we need to go through every (i, j) entry, adjust the column
        # by one (excluding the current solution), and then add that
        # possibility to our neighborhood
        neighborhood_list = []
        for (i, val) in enumerate(i_idx):
            orig_j_idx = j_idx[i]
            for j in range(self.n_label):
                if j == orig_j_idx:
                    continue
                else:
                    new_map = np.copy(z)
                    new_map[val, orig_j_idx] = 0
                    new_map[val, j] = 1
                    neighborhood_list.append(new_map)
        return np.array(neighborhood_list)

    def gen_one_step_neighborhood(self, z):
        """Generates a neighborhood where the difference moving one entry
        one column to the left or right

        Parameters
        ----------
        z: array, shape=(n_class, n_label)

        Returns
        -------
        array: neighborhood_list

        """

        # Get the necessary indexes
        i_idx, j_idx = self.find_combo_cols_idx(z)

        # Generate the neighborhood by considering all of the relevant
        # indexes and then either shifting the column by one to the left or
        # right
        neighborhood_list = []
        for i_val, j_val in zip(i_idx, j_idx):
            for j in [-1, 1]:
                col_val = j_val + j
                if col_val <= -1 or col_val >= self.n_label:
                    continue
                else:
                    new_map = np.copy(z)
                    new_map[i_val, j_val] = 0
                    new_map[i_val, col_val] = 1
                    neighborhood_list.append(new_map)
        return np.array(neighborhood_list)

    def check_dominated_solns(self, neighbor_list):
        """Checks if any of the provided neighbors are previously dominated
        solutions and removes them to improve computational performance

        Parameters
        ----------
        neighbor_list: array
            List of neighbors for the current best solution z

        Returns
        -------
        array: neighbor_list

        """

        # Find out all of the dominated solutions in the neighbor_list
        non_dominated_solns = [True] * len(neighbor_list)
        for (i, neighbor) in enumerate(neighbor_list):
            for dominated_soln in self.dominated_solns:
                if np.allclose(neighbor, dominated_soln):
                    non_dominated_solns[i] = False

        # Remove the dominated solutions from the list
        return neighbor_list[non_dominated_solns]

    def single_search(self, z):
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

        # Compute the objective value for the starting solution
        z_best = np.copy(z)
        obj_val = self.compute_objective(z)
        n_iter = 0

        # Continuously loop until the solution that we have is better
        # than any of the values in N(z)
        while True:
            # Generate the neighborhood around the current best z
            if self.neighborhood_type == 'complete':
                z_neighborhood = self.gen_complete_neighborhood(z_best)
            elif self.neighborhood_type == 'one_step':
                z_neighborhood = self.gen_one_step_neighborhood(z_best)
            else:
                z_neighborhood = self.gen_stochastic_neighborhood(z_best)

            # Remove any dominated solutions
            if len(self.dominated_solns) >= 1:
                z_neighborhood = self.check_dominated_solns(z_neighborhood)

            # Compute the objective value for each entry in N(z)
            new_obj_vals = [0.] * len(z_neighborhood)
            for (i, neighbor) in enumerate(z_neighborhood):
                new_obj_vals[i] = self.compute_objective(neighbor)
            if max(new_obj_vals) > obj_val:
                # Update the current best solution
                best_soln_idx = np.argmax(new_obj_vals)
                obj_val = max(new_obj_vals)
                z_best = np.copy(z_neighborhood[best_soln_idx])
                n_iter += 1

                # Update the list of dominated solutions to improve future
                # computation time
                for i in range(len(z_neighborhood)):
                    if i == best_soln_idx:
                        continue
                    else:
                        self.dominated_solns.append(z_neighborhood[i])
            else:
                # Stop; you've found the local optima
                break
        return {'z_best': z_best, 'obj_val': obj_val, 'n_iter': n_iter}

    def search(self):
        """Performs local search n_init times and finds the best search
        given the provided starting point

        Returns
        -------
        object: self

        """

        # Before we do anything we need to check if the provided solution
        # is already optimal; if so, return the solution, compute its
        # objective, and inform the user the provided solution was optimal
        if self.check_init_soln_is_opt():
            print('Provided solution was already optimal')
            return self.init_soln, self.compute_objective(self.init_soln)

        # Assuming the provided solution was not already optimal, we need
        # to generate n_init starting solutions
        with Pool() as p:
            starting_solns = p.map(self.gen_feasible_soln_garbage_fn,
                                   range(self.n_init))

        # Remove infeasible starting points
        starting_solns = self.remove_infeasible_solutions(starting_solns)

        # Using our starting solutions get the best solution for each
        # of them
        with Pool() as p:
            best_local_solns = p.map(self.single_search, starting_solns)

        # Grab the objective values from each of the solutions, determine
        # which one is the best, and then return the solution and value
        # which correspond to it
        obj_vals = [0.] * self.n_init
        for i in range(self.n_init):
            obj_vals[i] = best_local_solns[i]['obj_val']

        best_soln = np.argmax(obj_vals)
        self.label_map_ = best_local_solns[best_soln]['z_best'].astype(int)
        self.obj_val_ = np.max(obj_vals)
        self.n_iter_ = best_local_solns[best_soln]['n_iter']
        return self
