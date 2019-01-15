import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from cython.parallel import prange

# Define types we will use throughout the script
ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DOUBLE _sq_euclidean(np.ndarray[DOUBLE, ndim=1] x,
                           np.ndarray[DOUBLE, ndim=1] y):
    """
    Computes the squared euclidean distance between x and y
    """

    cdef:
        int i
        int n = x.shape[0]
        DOUBLE dist = 0.0

    # Go through each element of the arrays to compute ||x-y||^2
    for i in prange(n, nogil=True):
        dist += (x[i] - y[i])**2

    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[INT, ndim=1] _get_best_start_centers(
    np.ndarray[DOUBLE, ndim=2] V, np.ndarray[DOUBLE, ndim=2] centers,
    INT nclasses):

    """
    Gets the best starting group assignments given the starting centroids
    """

    cdef:
        np.ndarray[INT, ndim=1] z = np.empty(shape=(nclasses,), dtype=np.int32)
        int i
        int j
        int nlabels = centers.shape[0]

        np.ndarray[DOUBLE, ndim=2] dist_vals = np.empty(
            shape=(nclasses, nlabels), dtype=np.float64
        )

    # Go through every label and centroid and determine which one is the best
    for i in range(nclasses):
        for j in range(nlabels):
            dist_vals[i, j] = _sq_euclidean(V[i, :], centers[j, :])

        # z_i^* = argmin {dist_vals[i, :]}
        z[i] = dist_vals[i, :].argmin()

    return z


cpdef _get_idx_change(np.ndarray[long long, ndim=1] idx_nmc_t, int label,
                      int omc, np.ndarray[INT, ndim=1] z):
    """
    Gets the label indices that were affected by `label` moving
    """

    cdef:
        np.ndarray[long long, ndim=1] idx_omc_t = np.where(z == omc)[0]
        np.ndarray[long long, ndim=1] idx_omc_tn
        np.ndarray[long long, ndim=1] idx_nmc_tn

    # Delete the `label` from idx_omc_t
    idx_omc_tn = np.delete(idx_omc_t, np.where(idx_omc_t == label))

    # Add `label` to idx_nmc_tn
    idx_nmc_tn = np.append(idx_nmc_t, [label])

    return idx_omc_t, idx_omc_tn, idx_nmc_tn


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _update_centroid(np.ndarray[DOUBLE, ndim=2] V,
                       np.ndarray[DOUBLE, ndim=2] mu, int label,
                       float m_omc_t, float m_nmc_t, int nmc, int omc):
    """
    Updates the centroids as a consequence of moving `label` into a new
    meta-class
    """

    cdef:
        np.ndarray[DOUBLE, ndim=1] mu_nmc_tn = mu[nmc, :].copy()
        np.ndarray[DOUBLE, ndim=1] mu_omc_tn = mu[omc, :].copy()

    mu_nmc_tn = ((m_nmc_t * mu_nmc_tn) + V[label, :]) / (m_nmc_t + 1.0)
    mu_omc_tn = ((m_omc_t * mu_omc_tn) + V[label, :]) / (m_omc_t - 1.0)
    return mu_omc_tn, mu_nmc_tn


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DOUBLE _obj_change(np.ndarray[DOUBLE, ndim=2] V,
                         np.ndarray[DOUBLE, ndim=1] mu,
                         np.ndarray[DOUBLE, ndim=1] label_vars,
                         np.ndarray[long long, ndim=1] labels):
    """
    Computes the objective function change
    """

    cdef:
        DOUBLE dist = 0.0
        int i
        int n = labels.shape[0]

    # If our label set only contains one label then we need to use the
    # variance of that particular label; again this acts as a mechanism to
    # attempt to discourage the model from putting everything into one
    # meta-class
    if n == 1:
        return label_vars[labels[0]]
    else:
        for i in range(n):
            dist += _sq_euclidean(V[labels[i], :], mu)

    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DOUBLE _compute_change(np.ndarray[DOUBLE, ndim=2] V,
                             np.ndarray[DOUBLE, ndim=1] label_vars,
                             np.ndarray[DOUBLE, ndim=1] mu_omc,
                             np.ndarray[DOUBLE, ndim=1] mu_nmc,
                             np.ndarray[long long, ndim=1] idx_omc,
                             np.ndarray[long long, ndim=1] idx_nmc):
    """
    Computes the effect of moving a label for either step t or t+1
    """
    cdef DOUBLE change = 0.0

    change += _obj_change(V, mu_omc, label_vars, idx_omc)
    change += _obj_change(V, mu_nmc, label_vars, idx_nmc)
    return change


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _run_search(np.ndarray[DOUBLE, ndim=2] V, np.ndarray[INT, ndim=1] z,
                  np.ndarray[DOUBLE, ndim=2] mu,
                  np.ndarray[DOUBLE, ndim=1] label_vars,
                  float tol, int max_iter, int seed):
    """
    Runs the coordinate descent search
    """

    cdef:
        DOUBLE obj_val = 0.0
        DOUBLE obj_val_old
        int c = V.shape[0]
        int l = mu.shape[0]
        np.ndarray[INT, ndim=1] metaclasses = np.arange(l, dtype=np.int32)
        np.ndarray[INT, ndim=1] labels = np.arange(c, dtype=np.int32)
        np.ndarray[INT, ndim=1] label_options
        int iter_num
        int i
        int j
        np.ndarray[long long, ndim=1] mc_idx
        np.ndarray[long long, ndim=1] idx_nmc_t
        np.ndarray[long long, ndim=1] idx_omc_t
        np.ndarray[long long, ndim=1] idx_omc_tn
        np.ndarray[long long, ndim=1] idx_nmc_tn
        float m_omc_t
        float m_nmc_t
        DOUBLE t_change
        DOUBLE tn_change
        np.ndarray[DOUBLE, ndim=1] mu_omc_tn
        np.ndarray[DOUBLE, ndim=1] mu_nmc_tn


    # Set the seed for reproducibility
    np.random.seed(seed)

    # Compute the starting objective value
    for i in range(l):
        mc_idx = np.where(z == i)[0]
        obj_val += _obj_change(V, mu[i, :], label_vars, mc_idx)

    # We will continue to search the space until we reach convergence
    # or we hit the max iteration limit
    for iter_num in range(max_iter):

        # We will randomly search through the meta-classes and the label
        # space
        obj_val_old = obj_val
        np.random.shuffle(metaclasses)
        for i in range(l):

            # We need the current set of labels in meta-class i so that we
            # do not search over them when going through the label space
            idx_nmc_t = np.where(z == i)[0]
            label_options = np.delete(labels, np.where(np.isin(labels,
                                                       idx_nmc_t)[0]))

            label_options = np.random.permutation(label_options)
            for j in label_options:
                omc = z[j]

                # If the meta-class only contains one element we cannot
                # move it
                if len(np.where(z == omc)[0]) == 1:
                    continue

                # Get the indices that were affected by the label `j` moving
                idx_omc_t, idx_omc_tn, idx_nmc_tn = _get_idx_change(
                    idx_nmc_t, j, omc, z
                )

                # We need to count the number of labels that have been assigned
                # to the old and new meta-class
                m_omc_t = float(len(idx_omc_t))
                m_nmc_t = float(len(idx_nmc_t))

                # Update the centroid as a consequence of moving label `j`
                mu_omc_tn, mu_nmc_tn = _update_centroid(V, mu, j, m_omc_t,
                                                        m_nmc_t, i, omc)

                # Use all of the values to help us determine if moving label
                # `j` will improve our objective
                t_change = _compute_change(
                    V, label_vars, mu[omc, :], mu[i, :], idx_omc_t, idx_nmc_t)

                tn_change = _compute_change(
                    V, label_vars, mu_omc_tn, mu_nmc_tn, idx_omc_tn,
                    idx_nmc_tn
                )

                # If tn_change < t_change then we know that this move
                # must have improved the objective function
                if tn_change < t_change:
                    z[j] = i

                    # Update the centroid matrix
                    mu[i, :] = mu_nmc_tn
                    mu[omc, :] = mu_omc_tn

                    # Update the objective value
                    obj_val += (tn_change - t_change)
                    break

        # Check if we have met our convergence condition
        # |obj_val - obj_val_old| <= tol
        if abs((obj_val - obj_val_old)) <= tol:
            break

    # We want the final assignment vector we inferred from the search
    # and the corresponding objective value
    return z, obj_val
