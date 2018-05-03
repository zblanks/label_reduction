import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import re


def initialize(X, k_range, n_boot):
    """Initializes the EM algorithm for label reduction

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        k_range (ndarray, shape = [k_range, ]): List of cluster values to
        search over
        n_boot (int): Number of bootstrap iterations to perform

    Returns:
        ndarray, shape = [n_samples, ]: List of initial label assignments
        ndarray, shape = [n_class, n_features]: Matrix of coef weights
    """

    # # Determine the best k value for our data
    # best_k = run_gap_stat(X=X, k_range=k_range, n_boot=n_boot)
    #
    # # Using the best_k cluster our data into their corresponding groups
    # kmeans = KMeans(n_clusters=best_k, random_state=17, n_jobs=-1)
    # z = kmeans.fit_predict(X=X)
    z = np.random.choice([0, 1], size=X.shape[0])

    # Using our new labels, fit a linear model to predict them using
    # X as our our input
    model = LogisticRegression(penalty='l1', class_weight='balanced',
                               random_state=17)
    model.fit(X=X, y=z)

    # Get our model coefficients
    W = get_coef(model=model)
    return z, W


def get_coef(model):
    """Get our model coefficients

    Args:
        model (object): Sklearn model object

    Returns:
        (ndarray, shape = [n_class, n_features]): Model weights
    """
    W = model.coef_
    b = (model.intercept_).reshape(-1, 1)
    W = np.concatenate((b, W), axis=1)
    return W


def compute_obj(X, W):
    """Computes our likelihood function given the provided labels and
    corresponding model weights

    Args:
        X (ndarray, shape = [n_samples, n_features]: Data
        W (ndarray, shape = [n_class, n_features]: Model weights

    Returns:
        float: Log-likelihood objective value
    """

    # For now we're assuming a probabilistic model, so we use the
    # probability computation for logistic regression -- this can change
    # later
    probs = compute_label_prob(X=X, W=W)
    probs = np.max(probs, axis=1, keepdims=True)
    return np.sum(np.log(probs))


def compute_label_prob(X, W):
    """Computes the probability of seeing a label given our model weights

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        W (ndarray, shape = [n_class, n_features]): Model weights

    Returns:
        ndarray, shape = [n_samples, n_class]: Probability of seeing a
        particular label given our weights
    """

    # Assuming a logistic regression model, we will compute the probability
    # of seeing a particular label
    int = np.ones(shape=(X.shape[0], 1)) # Added for the B0 weight
    X = np.concatenate((int, X), axis=1)
    res = np.matmul(a=X, b=W.T)
    prob = 1 / (1 + np.exp(res))

    # Check for the binary case
    if prob.shape[1] == 1:
        prob = np.concatenate((prob, 1 - prob), axis=1)
    return prob


def train_model(X, Z):
    """Trains our model using our newly inferred labels and the data

    Args:
        X (ndarray, shape = [n_samples, n_features]): Data
        Z (ndarray, shape = [n_samples, n_class]): Label probabilities

    Returns:
        ndarray, shape = [n_class, n_features]: Learned model weights
    """

    # For now we're just going to use the label arg-max strategy
    z = np.argmax(Z, axis=1)

    # Train the model using our new labels
    model = LogisticRegression(penalty='l1', class_weight='balanced',
                               random_state=17)

    try:
        model.fit(X=X, y=z)
    except ValueError as e:
        print(e)
        W = 'did not train'
        return W

    W = get_coef(model=model)
    return W


def check_convergence(X, W, prev_loss, tol):
    """Checks if our algorithm has converged

    Args:
        X (ndarray, shape = [n_samples, n_features]: Data
        W (ndarray, shape = [n_class, n_features]: Model weights
        prev_loss (float): Log-likelihood value at stage t - 1
        tol (float): Tolerance to check for convergence

    Returns:
        bool: T/F of whether the algorithm has converged
        float: Log-likelihood for stage t
    """

    # Compute the log-likelihood for stage t
    curr_loss = compute_obj(X=X, W=W)

    # Check if we have converged
    if np.abs(curr_loss - prev_loss) < tol:
        return curr_loss, True
    else:
        return curr_loss, False


def run_em(X, k_range=np.arange(2, 10), n_boot=100, tol=1e-4, max_iter=100):
    """Runs our label reduction EM algorithm

    Args:
        X (ndarray, shape = [n_samples, n_features]: Data
        k_range (ndarray, shape = [k_range, ]): List of cluster values to
        search over
        n_boot (int): Number of bootstrap iterations to perform
        tol (float): Tolerance to check for EM convergence
        max_iter (int): Maximum number of iterations to allow the EM algorithm
                        to run

    Returns:
        ndarray, shape = [n_class, n_features]: Weights matrix
        list: Log-likelihood values for each iteration
        float: Final log-likelihood value
    """

    # Define a weights matrix to hold our results after each iteration
    w_list = []

    # Initialize our EM algorithm
    z, W = initialize(X=X, k_range=k_range, n_boot=n_boot)
    w_list.append(W)

    # Compute our initial loss value
    loss = compute_obj(X=X, W=W)

    # Hold our log-likelihood values to make sure the EM algorithm is
    # working as expected
    likelihood_vals = []
    likelihood_vals.append(loss)

    # Loop until we either converge or fail to converge
    for i in range(max_iter):

        # E-step
        Z = compute_label_prob(X=X, W=W)

        # M-step
        W = train_model(X=X, Z=Z)
        if isinstance(W, str) is True:
            return w_list, likelihood_vals, -np.inf
        else:
            w_list.append(W)

        # Compute the loss
        prev_loss = loss
        loss, has_converged = check_convergence(X=X, W=W, prev_loss=prev_loss,
                                                tol=tol)

        # Append to our loss list
        likelihood_vals.append(loss)

        # Check if we converged
        if has_converged is True:
            w_list = np.concatenate(w_list)
            likelihood_vals = np.array(likelihood_vals)
            return w_list, likelihood_vals, likelihood_vals[-1]

    # Let the user know that we failed to converge
    print('Failed to converge. Consider increasing the number of iterations')
    return W, likelihood_vals, likelihood_vals[-1]


def random_restart(X, n_init):
    """Runs multiple random instances of our algorithm and chooses
    the results with the best likelihood

    Args:
        X (ndarray, shape = [n_samples, n_features]: Data
        n_init (int): Number of random initializations of the algorithm

    Returns:
        ndarray, shape = [n_class, n_features]: Weights matrix
        list: Log-likelihood values for each iteration
    """

    # # Run our EM algorithm n_init times
    # with Pool() as p:
    #     weight_list, likelihood_list, final_likelihood = p.map(run_em,
    #                                                            [X] * n_init)

    weight_list = [None] * n_init
    likelihood_list = [None] * n_init
    final_likelihood = [None] * n_init
    for i in range(n_init):
        weight_list[i], likelihood_list[i], final_likelihood[i] = run_em(X)

    # Determine which iteration was the best
    best_model = np.argmax(final_likelihood)
    best_weights = weight_list[best_model]
    best_likelihood = likelihood_list[best_model]
    return best_weights, best_likelihood


if __name__ == '__main__':
    np.random.seed(17)

    # Get our working directory to save typing
    wd = 'C:/Users/zqb0731/Documents/label_reduction/synthetic_experiments'

    # Get all of our synthetic data
    files = os.listdir(os.path.join(wd, 'data'))

    for file in files:
        data = pd.read_csv(os.path.join(wd, 'data', file))
        data = data.as_matrix()

        # Run our algorithm
        weights, likelihood = random_restart(X=data, n_init=10)

        # Save the results to disk
        np.savetxt(os.path.join(wd, 'results',
                                re.sub('\.csv$', '', file) + '_weights.csv'),
                   X=weights, delimiter=',')
        np.savetxt(os.path.join(wd, 'results',
                                re.sub('\.csv$', '', file) + '_likelihood.csv'),
                   X=likelihood, delimiter=',')
