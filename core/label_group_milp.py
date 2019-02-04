from gurobipy import *
import numpy as np
from numba import njit
from joblib import Parallel, delayed


def _get_upperbound(V: np.ndarray):
    """
    Gets the M bound vector for the MILP
    """
    C, p = V.shape
    m_vals = np.empty_like(V)

    # Get the min and max for each feature
    l_vec = V.min(axis=0)
    u_vec = V.max(axis=0)

    # Go through each label and feature and determine the appropriate
    # upper bound
    for i in range(C):
        for k in range(p):
            m_vals[i, k] = max(abs(V[i, k] - l_vec[k]),
                               abs(V[i, k] - u_vec[k]))

    # Finally the max distance is the sum of the max distances along
    # each feature
    return m_vals.sum(axis=1)


def _label_group_milp(V: np.ndarray, M: np.ndarray, L: int):
    """
    Implements the MILP for grouping labels
    """

    # Get the number of labels and features in the data
    C, p = V.shape

    # Define the model
    m = Model()
    m.setParam('OutputFlag', False)

    # Add the variables
    z = m.addVars(C, L, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0) # LP relaxation
    mu = m.addVars(L, p, lb=-GRB.INFINITY)
    tau = m.addVars(C, L, p, lb=-GRB.INFINITY)
    gamma = m.addVars(C, L, lb=-GRB.INFINITY)
    delta = m.addVars(C, L)

    # Add the clustering constraints
    m.addConstrs(z.sum(i, '*') == 1 for i in range(C))
    m.addConstrs(z.sum('*', j) >= 1 for j in range(L))

    # Absolute value linearization constraints
    m.addConstrs(
        V[i, k] - mu[j, k] <= tau[i, j, k]
        for i in range(C) for j in range(L) for k in range(p)
    )

    m.addConstrs(
        mu[j, k] - V[i, k] <= tau[i, j, k]
        for i in range(C) for j in range(L) for k in range(p)
    )

    m.addConstrs(
        gamma[i, j] == tau.sum(i, j, '*')
        for i in range(C) for j in range(L)
    )

    # Objective linearization constraints
    m.addConstrs(delta[i, j] <= M[i] * z[i, j] for i in range(C) for j in range(L))
    m.addConstrs(delta[i, j] <= gamma[i, j] for i in range(C) for j in range(L))
    m.addConstrs(
        delta[i, j] >= gamma[i, j] - (M[i] * (1 - z[i, j]))
        for i in range(C) for j in range(L)
    )

    # Define the final objective function
    m.setObjective(delta.sum(), GRB.MINIMIZE)
    m.optimize()

    # Get the Z matrix to implement the sampling heuristic
    z_opt = m.getAttr('x', z)

    # Convert them into NumPy arrays so we can work with them more easily
    z_final = np.empty(shape=(C, L))
    for i in range(C):
        for j in range(L):
            z_final[i, j] = z_opt[i, j]

    return z_final


def _gen_soln(Z: np.ndarray, rng: np.random.RandomState):
    """
    Generates feasible integer solutions using a probabilistic sampling
    heuristic
    """
    Z_int = np.zeros(shape=Z.shape, dtype=np.int32)
    C, L = Z.shape

    # Go through each label, and sample proportional the "probability
    # distribution" defined by each row
    for i in range(C):
        group = rng.choice(a=np.arange(L), size=1, p=Z[i, :])
        Z_int[i, group] = 1

    return Z_int


def _check_valid_soln(Z: np.ndarray):
    """
    Checks if the sampled solution is valid by checking if each group
    has >= 1 label
    """
    if len(np.where(Z.sum(axis=0) == 0)[0]) > 0:
        return False
    else:
        return True


def _infer_centroid(V: np.ndarray, Z: np.ndarray):
    """
    Infers the centroid from the data and integer label map
    """
    L = Z.shape[1]
    p = V.shape[1]
    mu = np.empty(shape=(L, p))
    for j in range(L):
        # Get the labels in the particular group
        idx = np.where(Z[:, j])[0]
        mu[j, :] = V[idx, :].mean(axis=0)

    return mu


@njit
def _compute_objective(V: np.ndarray, Z: np.ndarray, mu: np.ndarray):
    """
    Computes the objective function from the proposed solution
    """
    obj = 0.0
    C, L = Z.shape
    for i in range(C):
        for j in range(L):
            obj += Z[i, j] * np.abs(V[i, :] - mu[j, :]).sum()

    return obj


def _run_heuristic(V: np.ndarray, Z: np.ndarray, rng: np.random.RandomState):
    """
    Runs the LP sampling heuristic and generate a final label map with its
    corresponding objective function
    """

    # First generate a solution
    Z_int = _gen_soln(Z, rng)

    # Check if the solution is valid; if it is not then we will pass an
    # empty matrix with an infinite objective function
    if not _check_valid_soln(Z_int):
        obj_val = np.inf
        return {'Z': Z_int, 'obj_val': obj_val}

    # Infer the centroid from the proposed label map
    mu = _infer_centroid(V, Z_int)

    # Finally compute the objective function from the decision variables
    obj_val = _compute_objective(V, Z_int, mu)
    return {'Z': Z_int, 'obj_val': obj_val}


def lp_heuristic(V: np.ndarray, k: int, niter=100):
    """
    Runs the LP heuristic and returns a final label map
    """

    # First get the M vector for the MILP
    M = _get_upperbound(V)

    # Run the LP relaxation of the MILP
    Z = _label_group_milp(V, M, k)

    # Using this starting LP solution generate niter feasible solutions and
    # get the best one
    rng_list = [np.random.RandomState(i) for i in range(niter)]
    with Parallel(n_jobs=-1) as p:
        res = p(delayed(_run_heuristic)(V, Z, rng) for rng in rng_list)

    # Determine the best solution
    obj_vals = np.array([res[i]['obj_val'] for i in range(niter)])
    best_soln = obj_vals.argmin()
    Z_best = res[best_soln]['Z']

    # Convert the matrix into the expected vector format for later analysis
    return np.nonzero(Z_best)[1]
